"""GPU, inference only. One model load scores, on the held-out rows of fold k, with the UNCHANGED public
native_q8 runtime (fp16, covering windows, token-weighted):
  slot 0 = continued state (alpha 1), slot 1 = public fold-k state, slot 2 = WiSE alpha 0.5 state.
Checks: public-runtime parity (public slot reproduces the author's OOF argmax >= 0.98) and export parity
(continued state agrees with the trainer's own first-window predictions on single-window rows >= 0.97).
Calibration stays the pinned public one. --smoke scores a few rows only (code/format test before training).
Works for v3 run directories (done.json) and the v2 fallback family prepared by wise.py (source.json)."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import BASE  # noqa: E402
from oof_targets import sha  # noqa: E402
import common as C  # noqa: E402

SLOTS = dict(ours=0, public=1, wise050=2)


def identity(run, k):
    """What is being evaluated, tied to verified bytes."""
    if (run / 'done.json').exists():
        done = json.loads((run / 'done.json').read_text())
        if done['args']['fold'] != k or done['method'] != 'public_native_oofce_v3':
            raise ValueError('Run is not the requested completed v3 fold')
        for name in ('trainables.safetensors', 'trainables_wise050.safetensors', 'val_probs_final.npy', 'val_rows.npy'):
            if sha(run / name) != done['artifact_sha256'][name]:
                raise ValueError(f'Changed v3 artifact: {name}')
        return dict(family='v3_oofce', done_sha256=sha(run / 'done.json'))
    source = json.loads((run / 'source.json').read_text())
    if source['fold'] != k or source['family'] != 'v2_grpo':
        raise ValueError('Fallback directory does not match the requested fold')
    if sha(run / 'trainables.safetensors') != source['state_sha256'] or sha(run / 'trainables_wise050.safetensors') != source['wise050_sha256']:
        raise ValueError('Changed fallback state')
    return dict(family='v2_grpo', source_sha256=sha(run / 'source.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--base', default=str(BASE / 'external/public_bases/qwen3-embedding-8b'))
    ap.add_argument('--mem-cap-gb', type=float, default=28.5); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--gpus', type=int, default=4, choices=[1, 4])
    a = ap.parse_args()
    run, k = a.run, a.fold
    who = identity(run, k)
    if not a.smoke:
        for name in ('native_eval.json', 'native_eval_parity_failed.json'):
            (run / name).unlink(missing_ok=True)
    import time
    import torch
    total = torch.cuda.mem_get_info(0)[1] / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb / total), 0)
    t0 = time.time()
    log = lambda *x: print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    mix = run / ('eval_assets_smoke' if a.smoke else 'eval_assets')
    (mix / 'fold0').mkdir(parents=True, exist_ok=True); (mix / 'fold2').mkdir(exist_ok=True)
    for name in ('manifest.json', 'base'):
        if not (mix / name).exists():
            os.symlink(C.NW / name, mix / name)
    for slot, filename in ((0, 'trainables.safetensors'), (2, 'trainables_wise050.safetensors')):
        link = mix / f'fold{slot}/trainables.safetensors'
        if not link.exists():
            os.symlink((run / filename).resolve(), link)
        if os.path.realpath(link) != str((run / filename).resolve()):
            raise ValueError(f'Stale evaluation slot {slot}')
    for slot, src in ((1, k), (3, 3), (4, 4)):
        if not (mix / f'fold{slot}').exists():
            os.symlink(C.NW / f'fold{src}', mix / f'fold{slot}')
    if os.path.realpath(mix / 'fold1') != os.path.realpath(C.NW / f'fold{k}'):
        raise ValueError('Public evaluation slot points at another fold')
    val = np.load(run / 'val_rows.npy')
    if set(val) != set(C.FOLDS[k]):
        raise ValueError('Validation rows are not public fold k')
    if a.smoke:
        val = val[:a.smoke]
    texts = C.tr.text.fillna('').astype(str).values[val].tolist()
    if a.gpus == 4:
        from parallel_native import ParallelNative
        model = ParallelNative(a.base, mix, run / 'native_work', log=log, gpus=4)
    else:
        model = C.native_q8.NativeQ8(a.base, mix, run / 'native_work', log=log, gpus=1)
    assert model.labels == C.classes
    res, finite = model.predict(texts, folds=[0, 1, 2], batch_size=a.batch)
    assert finite.all(), int((~finite).sum())
    P, Cn, B = res['plain'], res['centered'], res['centered_bias']
    pp, pc, _ = C.public_native_oof()
    pub_agree = float((P[1].argmax(1) == pp[val].argmax(1)).mean())
    seqs, owners, _ = C.native_q8.windows(model.tokenizer, texts)
    nwin = np.bincount(owners, minlength=len(texts))
    trainer = np.load(run / 'val_probs_final.npy')[:len(val)]
    single = nwin == 1
    own_agree = float((P[0][single].argmax(1) == trainer[single].argmax(1)).mean())
    y = C.y_all
    rec = dict(fold=k, **who, rows_val=int(len(val)), single_window_val_rows=int(single.sum()),
               public_runtime_argmax_agreement=round(pub_agree, 4),
               public_runtime_max_abs_logit_diff=round(float(np.abs(P[1] - pp[val]).max()), 4),
               export_argmax_agreement_single_window=round(own_agree, 4),
               plain_ba={name: round(100 * C.ba(y[val], P[slot].argmax(1)), 2) for name, slot in SLOTS.items()},
               seconds=round(time.time() - t0, 1))
    rec['parity_ok'] = bool(pub_agree >= .98 and own_agree >= .97)
    log(json.dumps(rec))
    if a.smoke:
        rec['smoke_ok'] = bool(pub_agree >= .98 and np.isfinite(P).all())
        json.dump(rec, open(run / ('smoke_ok.json' if rec['smoke_ok'] else 'smoke_failed.json'), 'w'), indent=1)
        sys.exit(0 if rec['smoke_ok'] else 1)
    np.savez(run / 'native_eval.npz', val=val, nwin_val=nwin,
             **{f'plain_{n}': P[s] for n, s in SLOTS.items()}, **{f'centered_{n}': Cn[s] for n, s in SLOTS.items()},
             **{f'bias_{n}': B[s] for n, s in SLOTS.items()})
    rec['artifact_sha256'] = {name: sha(run / name) for name in
                              ['trainables.safetensors', 'trainables_wise050.safetensors', 'native_eval.npz', 'val_rows.npy']}
    json.dump(rec, open(run / ('native_eval.json' if rec['parity_ok'] else 'native_eval_parity_failed.json'), 'w'), indent=1)
    sys.exit(0 if rec['parity_ok'] else 1)


if __name__ == '__main__':
    main()
