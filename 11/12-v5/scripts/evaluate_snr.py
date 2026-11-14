"""CPU, saved OOF only: nested prompt-fold evidence for SNR and structural controls, under the 0.684 deploy rule.
Candidate (fixed before evaluation): SNR = sibling-structured D + bias. Controls: bias only, diagonal only
(vector scaling), full 18x18 matrix. Reports native-alone BA/NLL, and the deploy rule at native share 0.40
(as deployed; proxy known to over-reward sharper native) and 0.75 (LB-anchored proxy), groups <= 2 (10 draws),
with prompt-bootstrap 95% CIs, plus groups <= 1 / <= 3 and mean native entropy (sharpness check)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared  # noqa: E402
import common as C  # noqa: E402
import proxy_v3 as X  # noqa: E402
import oof_targets as T  # noqa: E402
import families as F  # noqa: E402
import neuron_rewrite as N  # noqa: E402

MASKS = {'snr_sibling': N.MASK, 'snr_sibling_entropy_matched': N.MASK, 'bias_only': np.zeros((18, 18), bool),
         'diag_only': np.eye(18, dtype=bool), 'temperature_only': 'temperature', 'full_matrix': np.ones((18, 18), bool)}


def temperature_edits(o):
    """Control: one NLL-fitted scalar on both heads (pure sharpening), nested like the others."""
    out = {}
    for k in range(5):
        rows = np.flatnonzero(o['fold'] != k)
        best = min(np.exp(np.linspace(np.log(.5), np.log(2.5), 41)),
                   key=lambda t: N.nll_rows(o, rows, {h: ((t - 1) * np.eye(18), np.zeros(18)) for h in ('plain', 'centered')}))
        out[k] = dict(edit={h: ((best - 1) * np.eye(18), np.zeros(18)) for h in ('plain', 'centered')}, lam=float(best))
    return out


def nested(o, mask, name=''):
    if isinstance(mask, str):
        return temperature_edits(o)
    original = N.MASK
    N.MASK = mask
    fit_head = N.fit_head
    N.fit_head = lambda z, s, y, lam, mask=mask: fit_head(z, s, y, lam, mask)
    try:
        return N.nested_edits(o, match_entropy=name.endswith('entropy_matched'))
    finally:
        N.MASK = original; N.fit_head = fit_head


def native_all(o, edits, groups):
    out = np.zeros((len(o['y']), 18))
    for k in range(5):
        rows = np.flatnonzero(o['fold'] == k)
        out[rows] = N.native_lp(o, rows, None if edits is None else edits[k]['edit'], groups)
    return out


def main():
    o = N.load_oof()
    parent, _, _ = T.load_verified(shared.V3 / 'cache/oof_targets.npz', shared.BASE / 'data/original/train.csv',
                                   shared.BASE / 'data/folds.json', C.N)
    prior = X.label_prior(o['y']); y = o['y']
    results, store = {}, {}
    variants = {'public': None}
    for name, mask in MASKS.items():
        variants[name] = nested(o, mask, name)
    for name, edits in variants.items():
        rec = {}
        for gmax in (1, 2, 3):
            nat_avg, fin = np.zeros((C.N, 18)), {s: np.zeros((C.N, 18)) for s in (.40, .75)}
            for draw in range(X.DRAWS):
                g = N.groups_m(o['pid'], gmax, 2000 + draw)
                nl = native_all(o, edits, g)
                nat_avg += np.exp(nl) / X.DRAWS
                for s in fin:
                    fin[s] += X.deploy(parent, nl, prior, share=s) / X.DRAWS
            store[(name, gmax)] = (nat_avg, fin)
            if gmax == 2:
                rec['native_ba'] = 100 * X.ba(y, nat_avg.argmax(1))
                rec['native_nll'] = float(-np.log(nat_avg[np.arange(C.N), y] + 1e-12).mean())
                rec['native_entropy'] = float(-(nat_avg * np.log(nat_avg + 1e-12)).sum(1).mean())
            for s in fin:
                rec[f'deploy_s{s}_g{gmax}'] = 100 * X.ba(y, fin[s].argmax(1))
        if edits is not None:
            rec['lams'] = {str(k): v['lam'] for k, v in edits.items()}
        results[name] = rec
    base_n, base_f = store[('public', 2)]
    for name in MASKS:
        nat, fin = store[(name, 2)]
        r = results[name]
        r['vs_public'] = {'native': X.paired_report(y, o['pid'], base_n, nat)}
        for s in (.40, .75):
            r['vs_public'][f'deploy_s{s}'] = X.paired_report(y, o['pid'], base_f[s], fin[s])
    out = Path(__file__).resolve().parents[1] / 'artifacts/snr_evidence.json'
    out.write_text(json.dumps(results, indent=1) + '\n')
    for name, r in results.items():
        line = {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}
        print(name, json.dumps(line))
        for key, rep in r.get('vs_public', {}).items():
            print('   ', key, round(rep['delta'], 2), [round(x, 2) for x in rep['ci95']], 'fixed', rep['fixed'], 'harmed', rep['harmed'])


if __name__ == '__main__':
    main()
