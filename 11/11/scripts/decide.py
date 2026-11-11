"""11/11 pre-registered decision (CPU, validation only: held-out official-train rows of the folds whose 11/11 state passed
native_eval). Proxy of the Kaggle native slot, as in the public infer.py:
    P = mean over native states i of softmax((1 - 0.40) * log P_parent + 0.40 * lp_i)
with P_parent = public LoRA fold-adapter OOF stack (q14v5 3 / q8v5 2 / ll8v5 1) and lp_i = native_q8.calibrate on test-like
groups of 2 answers per prompt (10 random groupings, probabilities averaged). Every row is held out for both the public
state and ours of its fold, so a 5-state Kaggle average is approximated by the clean state of the row's fold:
    public_only : public state                                  (Kaggle: 5 public states)
    ours2       : 0.6 * public + 0.4 * ours                     (Kaggle: our folds 0,1 replace public folds 0,1)
    ours4       : 0.2 * public + 0.8 * ours                     (Kaggle: our folds 0-3 replace public folds 0-3)
ours2 needs >= 2 passing folds (the first two passing folds are used), ours4 needs all 4. The candidate with the largest
delta BA against public_only is adopted iff delta > 0 AND the prompt-bootstrap 95 % interval lower bound > 0 AND our state
used ALONE in the slot (ours_replace) is not worse than the public state (delta >= 0). The last guard was added on
2026-11-11 before any 11/11 training (CPU test with a degraded fake state: the blend proxy rewards plain state diversity
that the 5-state public average on Kaggle already has). Otherwise public_only. The test set and the leaderboard are
never used."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                         # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=str(C.R / 'runs')); ap.add_argument('--out', default=str(C.R / 'runs/decision.json'))
a = ap.parse_args(); RUNS = Path(a.runs)
SHARE, M, DRAWS = 0.40, 2, 10
y = C.y_all
parent = C.public_lora_stack(); pp, pc, pfold = C.public_native_oof()
folds_ok, per_fold = [], {}
for k in range(4):
    e = RUNS / f'fold{k}' / 'native_eval.json'
    if e.exists() and json.load(open(e)).get('parity_ok') and (RUNS / f'fold{k}' / 'calibration/fold.json').exists():
        folds_ok.append(k)
rows = np.concatenate([C.FOLDS[k] for k in folds_ok]) if folds_ok else np.array([], int)
out = dict(rule=__doc__.strip(), folds_passing=folds_ok, n_rows=int(len(rows)), split='held-out official train rows (VALIDATION, not test)')
if len(folds_ok) < 2:
    out.update(decision='public_only', reason=f'only {len(folds_ok)} passing fold(s): no candidate is defined')
    json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1)); sys.exit(0)


def mixture(lp_parent, lp_native):
    return np.exp(C.lsm((1 - SHARE) * lp_parent + SHARE * lp_native))


Ppub = np.zeros((len(rows), 18)); Pour = np.zeros((len(rows), 18)); alone = {}
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
cands = {'public_only': Ppub, 'ours2': 0.6 * Ppub + 0.4 * Pour}
if len(folds_ok) == 4:
    cands['ours4'] = 0.2 * Ppub + 0.8 * Pour
cands['ours_replace'] = Pour                                               # guard only, not deployable
yr = y[rows]; g = C.pidx[rows]; ug = np.unique(g); idx = [np.where(g == q)[0] for q in ug]; rng = np.random.default_rng(0)
boot = [np.concatenate([idx[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(2000)]
p0 = cands['public_only'].argmax(1); res = {}
for name, P in cands.items():
    p1 = P.argmax(1); d = np.array([C.ba(yr[b], p1[b]) - C.ba(yr[b], p0[b]) for b in boot]) if name != 'public_only' else np.zeros(1)
    res[name] = dict(ba=round(100 * C.ba(yr, p1), 2), delta=round(100 * (C.ba(yr, p1) - C.ba(yr, p0)), 2),
                     fixed=int(((p1 == yr) & (p0 != yr)).sum()), harmed=int(((p1 != yr) & (p0 == yr)).sum()),
                     ci95=[round(100 * float(np.quantile(d, q)), 2) for q in (0.025, 0.975)])
out['candidates'] = res
best = max([n for n in res if n not in ('public_only', 'ours_replace')], key=lambda n: res[n]['delta'])
guard = res['ours_replace']['delta'] >= 0
ok = res[best]['delta'] > 0 and res[best]['ci95'][0] > 0 and guard
out['guard_ours_replace_not_worse'] = bool(guard)
out['best_candidate'] = best
out['decision'] = best if ok else 'public_only'
out['reason'] = (f'{best}: delta {res[best]["delta"]:+.2f} BA, 95% CI {res[best]["ci95"]} excludes 0; ours_replace delta {res["ours_replace"]["delta"]:+.2f} >= 0' if ok else
                 f'best candidate {best}: delta {res[best]["delta"]:+.2f} BA, 95% CI {res[best]["ci95"]}; ours_replace delta {res["ours_replace"]["delta"]:+.2f} -> rule not met -> public_only')
out['our_folds_used'] = folds_ok[:2] if out['decision'] == 'ours2' else (folds_ok if out['decision'] == 'ours4' else [])
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))
