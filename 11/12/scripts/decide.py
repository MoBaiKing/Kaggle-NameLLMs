"""Fixed two/four-fold OOF proxy; public fallback when either safety guard fails."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import ROOT
from selection import CANDIDATES, assess, choose
from windows import sha
import common as C                                                         # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=str(ROOT / 'runs')); ap.add_argument('--out', default=str(ROOT / 'runs/decision.json'))
a = ap.parse_args(); RUNS = Path(a.runs)
SHARE, M, DRAWS = 0.40, 2, 10
y = C.y_all
parent = C.public_lora_stack(); pp, pc, pfold = C.public_native_oof()
folds_ok, per_fold = [], {}
for k in range(4):
    e = RUNS / f'fold{k}' / 'native_eval.json'
    if e.exists() and (RUNS / f'fold{k}' / 'calibration/fold.json').exists():
        check = json.load(open(e))
        if check.get('parity_ok'):
            if not check.get('artifact_sha256'):
                raise ValueError('Evaluation has no artifact fingerprint')
            for filename, expected in check['artifact_sha256'].items():
                if sha(RUNS / f'fold{k}' / filename) != expected:
                    raise ValueError(f'Changed evaluated artifact: fold{k}/{filename}')
            folds_ok.append(k)
rows = np.concatenate([C.FOLDS[k] for k in folds_ok]) if folds_ok else np.array([], int)
out = dict(rule=__doc__.strip(), folds_passing=folds_ok, n_rows=int(len(rows)), split='held-out official train rows (VALIDATION, not test)')
if not {0, 1}.issubset(folds_ok):
    out.update(decision='public_only', reason='Fixed folds 0 and 1 have not both passed parity')
    json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1)); sys.exit(0)


def mixture(lp_parent, lp_native):
    return np.exp(C.lsm((1 - SHARE) * lp_parent + SHARE * lp_native))


Ppub = np.zeros((len(rows), 18)); Pour = np.zeros((len(rows), 18))
pos = {int(r): i for i, r in enumerate(rows)}
for k in folds_ok:
    z = np.load(RUNS / f'fold{k}' / 'native_eval.npz'); val = z['val']; assert set(val) == set(C.FOLDS[k])
    st_o = json.load(open(RUNS / f'fold{k}' / 'calibration/fold.json'))['models']['qwen8']; st_p = C.public_state(k); bp = C.public_bias(k)
    ix = np.array([pos[int(r)] for r in val]); a_o, a_p = [], []
    for d in range(DRAWS):
        g = C.groups_sized(C.pid[val], M, M, 1000 + d)
        lo = C.state_lp(z['plain_ours'], z['centered_ours'], z['bias_ours'], st_o, g)
        lpub = C.state_lp(pp[val], pc[val], bp, st_p, g)
        Pour[ix] += mixture(parent[val], lo) / DRAWS; Ppub[ix] += mixture(parent[val], lpub) / DRAWS
        a_o.append(C.ba(y[val], lo.argmax(1))); a_p.append(C.ba(y[val], lpub.argmax(1)))
    per_fold[k] = dict(native_alone_ours=round(100 * np.mean(a_o), 2), native_alone_public=round(100 * np.mean(a_p), 2),
                       stack_ours=round(100 * C.ba(y[val], Pour[ix].argmax(1)), 2), stack_public=round(100 * C.ba(y[val], Ppub[ix].argmax(1)), 2),
                       calibration_ours={kk: st_o[kk] for kk in ('plain_scale', 'centered_scale', 'selected_centered_weight')})
out['per_fold'] = per_fold
reports = {}
for name, required, fraction in CANDIDATES:
    if not set(required).issubset(folds_ok):
        continue
    # A two-fold candidate uses only its own held-out folds, even when four
    # models are available. No undeployed fold enters its selection proxy.
    mask = np.isin(pfold[rows], required)
    reports[name] = assess(y[rows][mask], C.pid[rows][mask], Ppub[mask], Pour[mask], fraction)
    reports[name]['folds'] = list(required)
out['candidates'] = reports
out['decision'], out['our_folds_used'] = choose(reports)
out['reason'] = 'Fixed OOF candidates; positive paired prompt-bootstrap interval and standalone guard required'
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))
