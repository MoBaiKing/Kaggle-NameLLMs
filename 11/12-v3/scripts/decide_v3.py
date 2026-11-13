"""Pre-registered gate on a deployment-faithful validation proxy (CPU; reads saved GPU outputs only).

All 2,699 official rows, each scored by the native state that never saw it (public or continued), fused with
the frozen LoRA OOF parent exactly as the 0.684 notebook does (proxy_v3.deploy: geometric 0.6/0.4, Sinkhorn
0.30 / target mix 0.35, test-like groups of <= 2 answers, 10 grouping draws). Candidates are compared with the
public native under the SAME rule, paired, with a prompt-cluster bootstrap.

Adoption order (fixed before training): v3_wise050, then the Option-A fallback v2_wise050, v2_full, else
public_only. A candidate passes iff delta > 0, the 95% lower bound > -0.30 BA points, all five folds passed
parity, and the native expert alone does not lose more than 0.50 BA (guards against over-fitting to the
three-member proxy parent, which is not the full public stack). v3_full is reported, never adopted.
This is a validation proxy on official training rows; it is not the hidden test or the leaderboard.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import BASE, ROOT  # noqa: E402
import proxy_v3 as X  # noqa: E402
import oof_targets as T  # noqa: E402

ORDER = ['v3_wise050', 'v2_wise050', 'v2_full']
REPORT_ONLY = ['v3_full']
DELTA_MIN, LOWER_MIN, STANDALONE_MIN = 0.0, -0.30, -0.50
VARIANT_FILE = {'full': ('ours', 'trainables.safetensors'), 'wise050': ('wise050', 'trainables_wise050.safetensors')}


def load_family(root, C, sha):
    """Per-row raw native outputs of one family on all held-out rows, or None if any fold is missing/failed."""
    out = {name: np.zeros((C.N, X.CLASSES)) for name in
           ('plain_ours', 'centered_ours', 'plain_wise050', 'centered_wise050', 'plain_public', 'centered_public')}
    bias = {name: {} for name in ('ours', 'wise050', 'public')}
    records, seen = {}, np.zeros(C.N, bool)
    for k in range(5):
        run = root / f'fold{k}'
        path = run / 'native_eval.json'
        if not path.exists():
            return None, f'fold{k}: no passed native evaluation'
        rec = json.loads(path.read_text())
        if not rec.get('parity_ok'):
            return None, f'fold{k}: parity failed'
        for name, expected in rec['artifact_sha256'].items():
            if sha(run / name) != expected:
                raise ValueError(f'{run}/{name} changed after evaluation')
        with np.load(run / 'native_eval.npz', allow_pickle=False) as z:
            val = z['val']
            if set(val) != set(C.FOLDS[k]):
                raise ValueError(f'{run}: evaluated rows are not fold {k}')
            for name in out:
                out[name][val] = z[name]
            for name in bias:
                bias[name][k] = z[f'bias_{name}']
        seen[val] = True
        records[str(k)] = dict(evaluation_sha256=sha(path), **{f'{v}_sha256': rec['artifact_sha256'][f]
                               for v, (_, f) in VARIANT_FILE.items()})
    if not seen.all():
        return None, 'held-out rows incomplete'
    return dict(arrays=out, bias=bias, records=records), None


def score(C, parent, prior, plain, centered, biases, states, group_max=X.GROUP_MAX, sinkhorn=X.SINKHORN):
    final, alone = np.zeros((C.N, X.CLASSES)), np.zeros((C.N, X.CLASSES))
    for draw in range(X.DRAWS):
        groups = X.test_like_groups(C.pid, group_max, 2000 + draw)
        native = X.native_rows(C.native_q8.calibrate, plain, centered, C_row_fold(C), biases, states, groups)
        final += X.deploy(parent, native, prior, sinkhorn=sinkhorn) / X.DRAWS
        alone += np.exp(native) / X.DRAWS
    return final, alone


def C_row_fold(C):
    fold = np.full(C.N, -1)
    for k, rows in C.FOLDS.items():
        fold[rows] = k
    return fold


def evaluate(families, C, parent, prior, states, settings):
    """settings: name -> (parent, group_max, sinkhorn). Returns probabilities[setting][candidate]."""
    out = {}
    for name, (par, gmax, sk) in settings.items():
        res = {}
        base = next(iter(families.values()))
        a = base['arrays']
        res['public'] = score(C, par, prior, a['plain_public'], a['centered_public'], base['bias']['public'], states, gmax, sk)
        for fam, data in families.items():
            for variant in ('full', 'wise050'):
                key, _ = VARIANT_FILE[variant]
                res[f'{fam}_{variant}'] = score(C, par, prior, data['arrays'][f'plain_{key}'],
                                                data['arrays'][f'centered_{key}'], data['bias'][key], states, gmax, sk)
        out[name] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT / 'runs_v3')
    p.add_argument('--fallback', type=Path, default=None, help='runs_v3/fallback_v2 prepared by wise.py (Option A)')
    p.add_argument('--targets', type=Path, default=T.DEFAULT)
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    import common as C
    out_path = a.out or a.runs / 'decision.json'
    parent, _, targets_meta = T.load_verified(a.targets, BASE / 'data/original/train.csv', BASE / 'data/folds.json', C.N)
    prior = X.label_prior(C.y_all)
    states = {k: C.public_state(k) for k in range(5)}
    families, missing = {}, {}
    for fam, root in (('v3', a.runs), ('v2', a.fallback)):
        if root is None:
            missing[fam] = 'not provided'
            continue
        data, why = load_family(root, C, T.sha)
        if data is None:
            missing[fam] = why
        else:
            families[fam] = data
    if 'v3' in families and 'v2' in families:
        gap = np.abs(families['v3']['arrays']['plain_public'] - families['v2']['arrays']['plain_public']).max()
        if gap > 1e-2:
            raise ValueError(f'Public slot differs between families ({gap:.4f}); runtimes are not comparable')
    if not families:
        raise SystemExit(f'No complete candidate family: {missing}')
    parent_t1 = X.lsm(parent / X.PARENT_TEMPERATURE)
    settings = {'primary': (parent, X.GROUP_MAX, X.SINKHORN), 'parent_T1.0': (parent_t1, X.GROUP_MAX, X.SINKHORN),
                'groups_max1': (parent, 1, X.SINKHORN), 'groups_max3': (parent, 3, X.SINKHORN),
                'no_sinkhorn': (parent, X.GROUP_MAX, 0.0)}
    probs = evaluate(families, C, parent, prior, states, settings)
    row_fold = C_row_fold(C)
    reports = {}
    for cand in [c for c in ORDER + REPORT_ONLY if c in probs['primary']]:
        final, alone = probs['primary'][cand]
        pub_final, pub_alone = probs['primary']['public']
        r = X.paired_report(C.y_all, C.pid, pub_final, final)
        r['standalone_native_delta'] = 100 * (X.ba(C.y_all, alone.argmax(1)) - X.ba(C.y_all, pub_alone.argmax(1)))
        r['per_fold_delta'] = {str(k): 100 * (X.ba(C.y_all[row_fold == k], final[row_fold == k].argmax(1))
                                             - X.ba(C.y_all[row_fold == k], pub_final[row_fold == k].argmax(1))) for k in range(5)}
        r['sensitivity_delta'] = {s: 100 * (X.ba(C.y_all, probs[s][cand][0].argmax(1)) - X.ba(C.y_all, probs[s]['public'][0].argmax(1)))
                                  for s in settings if s != 'primary'}
        r['passed'] = bool(cand in ORDER and r['delta'] > DELTA_MIN and r['ci95'][0] > LOWER_MIN
                           and r['standalone_native_delta'] >= STANDALONE_MIN)
        reports[cand] = r
    decision = next((c for c in ORDER if reports.get(c, {}).get('passed')), 'public_only')
    chosen = None
    if decision != 'public_only':
        fam, variant = decision.split('_', 1)
        root = a.runs if fam == 'v3' else a.fallback
        chosen = dict(family=fam, variant=variant, filename=VARIANT_FILE[variant][1], root=str(root.resolve()),
                      folds=families[fam]['records'])
    out = dict(decision=decision, method='public_native_oofce_v3', chosen=chosen,
               rule=dict(order=ORDER, report_only=REPORT_ONLY, delta_min=DELTA_MIN, ci95_lower_min=LOWER_MIN,
                         standalone_native_delta_min=STANDALONE_MIN),
               proxy=dict(parent='three public LoRA members, label-free OOF fold mapping, T=1.75',
                          native_share=X.NATIVE_SHARE, sinkhorn=X.SINKHORN, target_mix=X.TARGET_MIX,
                          group_max=X.GROUP_MAX, draws=X.DRAWS, public_proxy_ba=reports and next(iter(reports.values()))['base_ba']),
               candidates=reports, missing_families=missing, targets_payload_sha256=targets_meta['payload_sha256'],
               split='official train rows, every native output held out for its state (VALIDATION proxy, not hidden test)',
               hidden_test_evaluated=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
