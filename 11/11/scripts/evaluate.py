"""GPU, inference only. For one trained fold state, run the UNCHANGED public native_q8.NativeQ8 (fp16, covering windows,
token-weighted) on the inner calibration rows and the held-out fold rows, for our state AND the public state of the same
fold (one model load). Checks:
  * public-runtime parity: the public state reproduces the author's published OOF (argmax agreement >= 0.98);
  * export parity: our exported state through native_q8 agrees with the trainer's own fold predictions on single-window
    rows (argmax agreement >= 0.97);
then fits our calibration on the inner rows (public calibration fields) and saves raw OOF outputs for decide.py.
--smoke: a few rows only, writes <run>/smoke_ok.json (code/format test before training)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='11/11/runs/fold<k> (or the smoke run dir)'); ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--base', default=str(C.R / 'external/public_bases/qwen3-embedding-8b'))
ap.add_argument('--mem-cap-gb', type=float, default=28.5); ap.add_argument('--smoke', type=int, default=0)
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--gpus', type=int, default=4, choices=[1,4])
a = ap.parse_args()
run = Path(a.run); k = a.fold
import torch                                                                  # noqa: E402
total = torch.cuda.mem_get_info(0)[1] / 2 ** 30
torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb / total), 0)
T0 = time.time()
log = lambda *x: print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)

# assets: slot 0 = our state, slot 1 = public state of fold k, slots 2-4 = public (only exist-checked by NativeQ8)
mix = run / 'eval_assets'; (mix / 'fold0').mkdir(parents=True, exist_ok=True)
for name in ('manifest.json', 'base'):
    if not (mix / name).exists():
        os.symlink(C.NW / name, mix / name)
ours = run / 'trainables.safetensors'; assert ours.exists(), ours
if not (mix / 'fold0/trainables.safetensors').exists():
    os.symlink(ours.resolve(), mix / 'fold0/trainables.safetensors')
for slot, src in ((1, k), (2, 2), (3, 3), (4, 4)):
    if not (mix / f'fold{slot}').exists():
        os.symlink(C.NW / f'fold{src}', mix / f'fold{slot}')

val = np.load(run / 'val_rows.npy'); inner = np.load(run / 'inner_rows.npy')
assert set(val) == set(C.FOLDS[k]) and not set(inner) & set(val)
if a.smoke:
    val, inner = val[:a.smoke], inner[:a.smoke // 2]
rows = np.concatenate([inner, val]); texts = C.tr.text.fillna('').astype(str).values[rows].tolist()
if a.gpus == 4:
    from parallel_native import ParallelNative
    model = ParallelNative(a.base, mix, run / 'native_work', log=log, gpus=4)
else:
    model = C.native_q8.NativeQ8(a.base, mix, run / 'native_work', log=log, gpus=1)
assert model.labels == C.classes
res, finite = model.predict(texts, folds=[0, 1], batch_size=a.batch)
assert finite.all(), int((~finite).sum())
P, Cn, B = res['plain'], res['centered'], res['centered_bias']                # [2, rows, 18], [2, 18]
nv = len(val); vi = slice(len(inner), None); ii = slice(0, len(inner))

# public-runtime parity on the held-out fold
pp, pc, pf = C.public_native_oof()
pub_agree = float((P[1][vi].argmax(1) == pp[val].argmax(1)).mean()); pub_maxdiff = float(np.abs(P[1][vi] - pp[val]).max())
# export parity: single-window rows, trainer's own (bf16, first window) predictions
seqs, owners, _ = C.native_q8.windows(model.tokenizer, texts); nwin = np.bincount(owners, minlength=len(texts))[vi]
tv = np.load(run / 'val_probs_final.npy')
if a.smoke:
    tv = tv[:nv]
single = nwin == 1
own_agree = float((P[0][vi][single].argmax(1) == tv[single].argmax(1)).mean())
y = C.y_all
rec = dict(fold=k, rows_val=int(nv), rows_inner=int(len(inner)), single_window_val_rows=int(single.sum()),
           public_runtime_argmax_agreement=round(pub_agree, 4), public_runtime_max_abs_logit_diff=round(pub_maxdiff, 4),
           export_argmax_agreement_single_window=round(own_agree, 4),
           ba_plain_ours_val=round(100 * C.ba(y[val], P[0][vi].argmax(1)), 2), ba_plain_public_val=round(100 * C.ba(y[val], P[1][vi].argmax(1)), 2),
           ba_plain_public_author_oof_val=round(100 * C.ba(y[val], pp[val].argmax(1)), 2), seconds=round(time.time() - T0, 1))
rec['parity_ok'] = bool(pub_agree >= 0.98 and own_agree >= 0.97)
log(json.dumps(rec))
if a.smoke:
    # smoke: an untrained state is near-constant, so export agreement is only informative if the trainer output is too;
    # the smoke requires the public-runtime parity and a clean load/predict of our exported state
    rec['smoke_ok'] = bool(pub_agree >= 0.98 and np.isfinite(P[0]).all())
    json.dump(rec, open(run / ('smoke_ok.json' if rec['smoke_ok'] else 'smoke_failed.json'), 'w'), indent=1)
    sys.exit(0 if rec['smoke_ok'] else 1)
cal = C.fit_calibration(P[0][ii], Cn[0][ii], B[0], y[inner], C.pid[inner])
rec['calibration'] = {kk: v for kk, v in cal.items() if kk != 'grid'}
(run / 'calibration').mkdir(exist_ok=True)
json.dump({'models': {'qwen8': cal}, 'source': '11/11 inner calibration rows (training fold, held out of both training stages)'},
          open(run / 'calibration/fold.json', 'w'), indent=1)
np.savez(run / 'native_eval.npz', val=val, inner=inner, plain_ours=P[0][vi], centered_ours=Cn[0][vi], bias_ours=B[0],
         plain_public=P[1][vi], centered_public=Cn[1][vi], bias_public=B[1], nwin_val=nwin)
json.dump(rec, open(run / ('native_eval.json' if rec['parity_ok'] else 'native_eval_parity_failed.json'), 'w'), indent=1)
sys.exit(0 if rec['parity_ok'] else 1)
