"""CPU only. Staged validation evidence for v4 on all 2,699 official rows and the final B (SAGE) bundle.

Stages, all under the deployment rule of the 0.684 notebook (Sinkhorn, groups <= 2, 10 draws) on the LB-anchored
proxy (proxy_v4: native share 0.75 in the proxy plays the role of the deployed 0.40):
  S0  public stack proxy with the public native            (reference only, never a submission)
  S1  + A1 (HSC-Native, WiSE alpha 0.5 states, held-out per fold)
  S2  S1 + B without A2       (B fitted by nested prompt-fold CV: outer = public folds, inner = remaining folds)
  S3  S1 + B with A2 = v4 full (A2 never saw official rows, so its scores are clean held-out evidence)
Paired prompt-cluster bootstrap CIs for S1-S0, S2-S1, S3-S1, S3-S0; A2 alone and top-K recall as diagnostics.
The submission candidate is always v4 full; these numbers are evidence (no public fallback).
The final B weights are refit on all rows with C chosen by the same CV and written to runs_v4/b/b_bundle.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import BASE, ROOT  # noqa: E402
import proxy_v3 as X  # noqa: E402
import oof_targets as T  # noqa: E402
import families as F  # noqa: E402
import sage as S  # noqa: E402
import external_v4 as E  # noqa: E402
import proxy_v4 as X4  # noqa: E402
import oof_targets_v4 as T4  # noqa: E402

CS = (.03, .1, .3, 1., 3.)


def stage_probs(C, parent, prior, plain, centered, biases, states, row_fold):
    final = np.zeros((C.N, X.CLASSES))
    for draw in range(X.DRAWS):
        groups = X.test_like_groups(C.pid, X.GROUP_MAX, 2000 + draw)
        nl = X.native_rows(C.native_q8.calibrate, plain, centered, row_fold, biases, states, groups)
        final += X4.deploy(parent, nl, prior) / X.DRAWS
    return final


def load_a1(root, C, sha):
    arrays = {n: np.zeros((C.N, X.CLASSES)) for n in ('plain_wise050', 'centered_wise050')}
    bias, records = {}, {}
    for k in range(5):
        run = root / f'fold{k}'
        rec = json.loads((run / 'native_eval.json').read_text())
        if not rec.get('parity_ok'):
            raise SystemExit(f'A1 fold{k} has not passed parity')
        for name, expected in rec['artifact_sha256'].items():
            if sha(run / name) != expected:
                raise ValueError(f'A1 fold{k}/{name} changed after evaluation')
        with np.load(run / 'native_eval.npz') as z:
            if set(z['val']) != set(C.FOLDS[k]):
                raise ValueError(f'A1 fold{k} evaluated other rows')
            for n in arrays:
                arrays[n][z['val']] = z[n]
            bias[k] = z['bias_wise050']
        records[str(k)] = dict(evaluation_sha256=sha(run / 'native_eval.json'),
                               wise050_sha256=rec['artifact_sha256']['trainables_wise050.safetensors'])
    return arrays, bias, records


def prepare(rows, final, typo, a2, neighbors):
    cand = S.candidates(final[rows])
    gated = S.gate(final[rows], cand, neighbors)
    sums = tokens = None
    if a2 is not None:
        sums = np.take_along_axis(a2[0][rows], cand, 1); tokens = a2[1][rows]
    Xm, _ = S.features(final[rows], typo[rows], cand, neighbors, sums, tokens)
    return Xm, cand, gated


def fit_rows(rows, y, final, typo, a2, neighbors, Cval):
    Xm, cand, gated = prepare(rows, final, typo, a2, neighbors)
    use, target = S.training_target(cand, gated, y[rows])
    return S.fit_logit(Xm[use], target[use], Cval)


def predict_rows(rows, w, final, typo, a2, neighbors):
    Xm, cand, gated = prepare(rows, final, typo, a2, neighbors)
    return S.decide(Xm, cand, gated, w), gated


def choose_C(rows, folds_of, y, final, typo, a2, neighbors):
    scores = {}
    for Cval in CS:
        pred = np.zeros(len(y), int)
        for j in np.unique(folds_of[rows]):
            tr, te = rows[folds_of[rows] != j], rows[folds_of[rows] == j]
            w = fit_rows(tr, y, final, typo, a2, neighbors, Cval)
            pred[te] = predict_rows(te, w, final, typo, a2, neighbors)[0]
        scores[Cval] = X.ba(y[rows], pred[rows])
    return max(scores, key=scores.get), scores


def nested(y, folds_of, final, typo, a2, neighbors):
    pred = np.zeros(len(y), int); gated_all = np.zeros(len(y), bool); chosen = {}
    for k in range(5):
        tr, te = np.flatnonzero(folds_of != k), np.flatnonzero(folds_of == k)
        Cval, _ = choose_C(tr, folds_of, y, final, typo, a2, neighbors)
        w = fit_rows(tr, y, final, typo, a2, neighbors, Cval)
        pred[te], gated_all[te] = predict_rows(te, w, final, typo, a2, neighbors)
        chosen[str(k)] = Cval
    return pred, gated_all, chosen


def onehot(pred):
    return np.eye(X.CLASSES)[pred]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a1', type=Path, default=ROOT / 'runs_v4/a1')
    p.add_argument('--a2', type=Path, default=ROOT / 'runs_v4/a2')
    p.add_argument('--external', type=Path, default=E.DEFAULT)
    p.add_argument('--targets', type=Path, default=T4.DEFAULT)
    p.add_argument('--out', type=Path, default=ROOT / 'runs_v4/b')
    a = p.parse_args()
    import common as C
    parent, _, tmeta = T.load_verified(a.targets, BASE / 'data/original/train.csv', BASE / 'data/folds.json', C.N)
    prior = X.label_prior(C.y_all); y = C.y_all
    states = {k: C.public_state(k) for k in range(5)}
    row_fold = np.full(C.N, -1)
    for k, rows in C.FOLDS.items():
        row_fold[rows] = k
    if C.classes != F.CLASSES:
        raise ValueError('Class order differs')
    neighbors = F.NEIGHBORS
    # S0 (reference) and S1 (A1)
    pp, pc, _ = C.public_native_oof()
    s0 = stage_probs(C, parent, prior, pp, pc, {k: C.public_bias(k) for k in range(5)}, states, row_fold)
    arrays, bias1, a1_records = load_a1(a.a1, C, T.sha)
    s1 = stage_probs(C, parent, prior, arrays['plain_wise050'], arrays['centered_wise050'], bias1, states, row_fold)
    # A2 clean held-out scores
    a2_meta = json.loads((a.a2 / 'a2_scores.json').read_text())
    with np.load(a.a2 / 'a2_scores.npz') as z:
        if T.sha(a.a2 / 'a2_scores.npz') != a2_meta['scores_sha256']:
            raise ValueError('A2 scores changed')
        a2 = (z['official_sums'], z['official_tokens'], z['official_done'])
    if not a2[2].all():
        raise SystemExit('A2 did not score every official row')
    # typography evidence from external data only
    ext = pd.read_parquet(a.external); ext = ext[ext.split == 'train_pool']
    table = S.fit_typography(ext.text.values, ext.label.values, X.CLASSES)
    typo = S.typography_lp(table, C.tr.text.fillna('').astype(str).values)
    # B under nested CV
    b0, g0, c0 = nested(y, row_fold, s1, typo, None, neighbors)
    b1, g1, c1 = nested(y, row_fold, s1, typo, a2, neighbors)
    rep = lambda base, cand: X.paired_report(y, C.pid, base, cand)
    cand1 = S.candidates(s1)
    reports = {'S1_A1_vs_S0': rep(s0, s1), 'S2_B_noA2_vs_S1': rep(s1, onehot(b0)), 'S3_full_vs_S1': rep(s1, onehot(b1)),
               'S3_full_vs_S0': rep(s0, onehot(b1))}
    diagnostics = dict(S0_ba=100 * X.ba(y, s0.argmax(1)), S1_ba=100 * X.ba(y, s1.argmax(1)),
                       S2_ba=100 * X.ba(y, b0), S3_ba=100 * X.ba(y, b1),
                       a2_alone_ba=a2_meta['official_ba_alone'], a2_dev_ba=a2_meta['dev_ba_alone'],
                       topk_recall_S1=100 * float((cand1 == y[:, None]).any(1).mean()),
                       gated_rows=int(g1.sum()), nested_C={'no_a2': c0, 'a2': c1},
                       typography_alone_ba=100 * X.ba(y, typo.argmax(1)))
    # final B on all rows
    rows = np.arange(C.N)
    Cn, _ = choose_C(rows, row_fold, y, s1, typo, None, neighbors)
    Ca, _ = choose_C(rows, row_fold, y, s1, typo, a2, neighbors)
    w_no = fit_rows(rows, y, s1, typo, None, neighbors, Cn)
    w_a2 = fit_rows(rows, y, s1, typo, a2, neighbors, Ca)
    a.out.mkdir(parents=True, exist_ok=True)
    bundle = dict(method='v4_b_sage', k=S.K, gate_q=S.GATE_Q, neighbors=neighbors.astype(int).tolist(),
                  typography=table, w_no_a2=w_no.tolist(), w_a2=w_a2.tolist(), C=dict(no_a2=Cn, a2=Ca),
                  features_a2=S.FEATURES_A2, features_no_a2=S.FEATURES_NO_A2, classes=F.CLASSES,
                  sage_sha256=T.sha(Path(S.__file__)), fitted_on='S1 proxy (A1 WiSE 0.5) on all 2,699 official rows')
    (a.out / 'b_bundle.json').write_text(json.dumps(bundle) + '\n')
    passed = lambda r: bool(r['delta'] > 0 and r['ci95'][0] > -.30)
    out = dict(decision='v4_full', method='v4_a1_a2_b', public_fallback=False,
               evidence={k: dict(v, supports=passed(v), warning=('clearly harmful' if v['ci95'][1] < 0 else None))
                         for k, v in reports.items()},
               diagnostics=diagnostics, a1=dict(root=str(a.a1.resolve()), folds=a1_records),
               a2=dict(root=str(a.a2.resolve()), scores=a2_meta), b=dict(bundle=str((a.out / 'b_bundle.json').resolve()),
                                                                          sha256=T.sha(a.out / 'b_bundle.json')),
               targets_payload_sha256=tmeta['payload_sha256'],
               split='official train rows; A1 per-fold held-out, A2 never trained on them, B nested prompt-fold CV (VALIDATION, not hidden test)',
               hidden_test_evaluated=False)
    (a.out / 'decision.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(evidence={k: (round(v['delta'], 2), [round(x, 2) for x in v['ci95']]) for k, v in reports.items()},
                          **{k: v for k, v in diagnostics.items() if k != 'nested_C'}), indent=1))


if __name__ == '__main__':
    main()
