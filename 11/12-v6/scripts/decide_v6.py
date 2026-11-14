"""CPU gate for v6 (reads saved GPU evaluation outputs only). The submission decision uses the native expert ALONE,
calibrated with each fold's own calibration, on all 2,699 official rows (every row scored by the state that never
saw it), against the public native's author OOF on the same rows. This is the metric that called the leaderboard
direction for both previous native changes (v4 A1 and v5.0 SNR), unlike the ensemble proxy.
Pass iff, with hidden-test-like groups of <= 2 answers per prompt (10 draws), delta > 0 and the 95% prompt-bootstrap
lower bound > 0, AND with singletons delta >= 0, AND all five folds passed public-runtime/export parity.
Pass -> decision 'v6' (the build may proceed); otherwise 'not_ready' (nothing is built; there is no public fallback).
The LB-anchored ensemble proxy is reported for information only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import ROOT  # noqa: E402
import common as C  # noqa: E402
import proxy_v3 as X  # noqa: E402
from windows import sha  # noqa: E402

DRAWS = 10


def groups_m(m, seed):
    rng = np.random.default_rng(seed); g = np.zeros(C.N, np.int64); n = 0
    for q in np.unique(C.pid):
        r = rng.permutation(np.flatnonzero(C.pid == q))
        for i in range(0, len(r), m):
            g[r[i:i + m]] = n; n += 1
    return g


def native_probs(plain, cent, bias, states, row_fold, groups):
    out = np.zeros((C.N, 18))
    for k in range(5):
        rows = np.flatnonzero(row_fold == k)
        out[rows] = np.exp(C.state_lp(plain[rows], cent[rows], bias[k], states[k], groups[rows]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT / 'runs_v6')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    out_path = a.out or a.runs / 'decision.json'
    pp, pc, row_fold = C.public_native_oof()
    plain, cent = np.zeros_like(pp), np.zeros_like(pc)
    ours_state, ours_bias, records, missing = {}, {}, {}, []
    for k in range(5):
        run = a.runs / f'fold{k}'
        path = run / 'native_eval.json'
        if not path.exists():
            missing.append(f'fold{k}: no passed native evaluation'); continue
        rec = json.loads(path.read_text())
        if not rec.get('parity_ok'):
            missing.append(f'fold{k}: parity failed'); continue
        for name, expected in rec['artifact_sha256'].items():
            if sha(run / name) != expected:
                raise ValueError(f'fold{k}/{name} changed after evaluation')
        with np.load(run / 'native_eval.npz', allow_pickle=False) as z:
            if set(z['val']) != set(C.FOLDS[k]):
                raise ValueError(f'fold{k} evaluated other rows')
            plain[z['val']], cent[z['val']] = z['plain_ours'], z['centered_ours']
            ours_bias[k] = z['bias_ours']
        ours_state[k] = json.loads((run / 'calibration/fold.json').read_text())['models']['qwen8']
        records[str(k)] = dict(trainables_sha256=rec['artifact_sha256']['trainables.safetensors'],
                               calibration_sha256=rec['artifact_sha256']['calibration/fold.json'],
                               evaluation_sha256=sha(path), parity=dict(public=rec['public_runtime_argmax_agreement'],
                                                                         export=rec['export_argmax_agreement_single_window']))
    if missing:
        decision = dict(decision='not_ready', reason=missing, public_fallback=False)
        out_path.write_text(json.dumps(decision, indent=2) + '\n'); print(json.dumps(decision, indent=2)); return
    pub_state = {k: C.public_state(k) for k in range(5)}; pub_bias = {k: C.public_bias(k) for k in range(5)}
    reports, probs = {}, {}
    for name, m in (('groups_max2', 2), ('singletons', 1), ('groups_max3', 3)):
        ours, pub = np.zeros((C.N, 18)), np.zeros((C.N, 18))
        for d in range(DRAWS if m > 1 else 1):
            g = groups_m(m, 4000 + d)
            ours += native_probs(plain, cent, ours_bias, ours_state, row_fold, g) / (DRAWS if m > 1 else 1)
            pub += native_probs(pp, pc, pub_bias, pub_state, row_fold, g) / (DRAWS if m > 1 else 1)
        reports[name] = X.paired_report(C.y_all, C.pid, pub, ours)
        reports[name]['per_fold_delta'] = {str(k): 100 * (X.ba(C.y_all[row_fold == k], ours[row_fold == k].argmax(1))
                                                         - X.ba(C.y_all[row_fold == k], pub[row_fold == k].argmax(1))) for k in range(5)}
        probs[name] = (ours, pub)
    g2, g1 = reports['groups_max2'], reports['singletons']
    passed = bool(g2['delta'] > 0 and g2['ci95'][0] > 0 and g1['delta'] >= 0)
    decision = dict(decision='v6' if passed else 'not_ready', method='v6_native_data_soup', public_fallback=False,
                    gate=dict(metric='native-alone balanced accuracy vs public native (author OOF), all 2,699 official rows',
                              rule='groups<=2: delta > 0 and 95% lower > 0; singletons: delta >= 0; five folds parity_ok',
                              passed=passed),
                    native_alone=reports, folds=records, root=str(a.runs.resolve()),
                    split='official train rows, each scored by the fold state that never saw it (VALIDATION, not the leaderboard)')
    out_path.write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps({k: (round(v['base_ba'], 2), round(v['ba'], 2), round(v['delta'], 2), [round(x, 2) for x in v['ci95']])
                      for k, v in reports.items()} | dict(decision=decision['decision']), indent=1))


if __name__ == '__main__':
    main()
