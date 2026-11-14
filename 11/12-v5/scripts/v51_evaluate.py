"""CPU gate: native-alone sparse-group BA, sibling accuracy, conditional family accuracy.
No three-member parent mixture enters any acceptance criterion. Official train OOF is validation.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v51_common import (BASE, ROOT, RUN, CLASSES, GROUPS, FAMILY, dump, sha, row_folds,
                        sparse_groups, paired_ci, conditional, ba, save_npz, load_verified)
import v51_eb as EB
import v51_sage as B
from v51_train import split_name


def expert_scores(root, group, excluded, official, fold, artifacts):
    path = Path(root) / group / split_name(excluded) / 'scores.npz'
    rec = load_verified(path)
    completion = path.parent / 'complete.json'
    if not completion.exists() or json.loads(completion.read_text()) != rec:
        raise ValueError('Missing or inconsistent expert completion record')
    if rec.get('smoke') or sorted(rec['excluded']) != sorted(excluded):
        raise ValueError('Smoke/incompatible expert scores cannot enter a gate')
    if rec['identity']['official_sha256'] != sha(BASE / 'data/original/train.csv') or rec['identity']['folds_sha256'] != sha(BASE / 'data/folds.json'):
        raise ValueError('Expert scores use different official data/folds')
    if rec['identity']['plan_sha256'] != sha(ROOT / 'artifacts/v51_plan.json'):
        raise ValueError('Expert scores use another frozen plan')
    if (rec['identity']['code_sha256'] != sha(ROOT / 'scripts/v51_train.py') or
            rec['identity']['expert_sha256'] != sha(ROOT / 'scripts/v51_expert.py')):
        raise ValueError('Expert code changed since training')
    checkpoint = path.parent / 'final.safetensors'
    if sha(checkpoint) != rec['checkpoint_sha256']:
        raise ValueError('Expert checkpoint changed')
    with np.load(path, allow_pickle=False) as z:
        rows, ids, lp, labels = z['rows'], z['ids'].astype(str), z['lp'], z['classes'].tolist()
    expected = np.flatnonzero(np.isin(fold, excluded))
    if not np.array_equal(rows, expected) or not np.array_equal(ids, official.id.iloc[rows].astype(str)) or labels != GROUPS[group]:
        raise ValueError('Expert OOF rows, IDs, exclusions or class order changed')
    if lp.shape != (len(rows), len(labels)) or not np.isfinite(lp).all():
        raise ValueError('Invalid expert scores')
    artifacts[str(path.resolve())] = sha(path)
    artifacts[str(path.with_suffix('.json').resolve())] = sha(path.with_suffix('.json'))
    artifacts[str(checkpoint.resolve())] = sha(checkpoint)
    artifacts[str(completion.resolve())] = sha(completion)
    out = np.full((len(official), 18), np.nan)
    out[np.ix_(rows, [CLASSES.index(c) for c in labels])] = lp
    return out


def nested_sage(probs, y, fold, group_scores, enabled, sage):
    pred = np.zeros(len(y), int); weights = {}
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        training = np.full((len(y), 18), np.nan)
        evaluation = np.full_like(training, np.nan)
        for group in enabled:
            labels = [CLASSES.index(c) for c in GROUPS[group]]
            evaluation[np.ix_(te, labels)] = group_scores[group][(k,)][np.ix_(te, labels)]
            for j in range(5):
                if j == k:
                    continue
                rows = np.flatnonzero(fold == j)
                training[np.ix_(rows, labels)] = group_scores[group][tuple(sorted((k, j)))][np.ix_(rows, labels)]
        w = B.fit(probs[tr], training[tr], enabled, y[tr], sage)
        pred[te] = B.apply(probs[te], evaluation[te], enabled, w, sage)
        weights[str(k)] = w.tolist()
    return pred, weights


def choose_components(eb_report, groups, combined=None):
    eb = bool(eb_report['ci95'][0] > 0)
    experts = [g for g in GROUPS if g in groups and groups[g]['within_delta'] > 0 and groups[g]['family']['ci95'][0] > 0]
    if experts and (combined is None or combined['ci95'][0] <= 0):
        experts = []
    return dict(eb=eb, experts=experts, build=bool(eb or experts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', type=Path, default=RUN / 'collect/features.npz')
    p.add_argument('--experts', type=Path, default=RUN / 'experts')
    p.add_argument('--out', type=Path, default=RUN / 'evidence')
    p.add_argument('--bootstrap', type=int, default=4000)
    p.add_argument('--eb-only', action='store_true', help='Evaluate completed EB without enabling unfinished experts')
    a = p.parse_args()
    rec = load_verified(a.collect)
    if rec.get('smoke'):
        raise ValueError('Smoke inference cannot be used for acceptance')
    if rec['identity']['code_sha256'] != sha(ROOT / 'scripts/v51_collect.py'):
        raise ValueError('Collector code changed since extraction')
    official = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    fold = row_folds(official, json.loads((BASE / 'data/folds.json').read_text()))
    y = np.searchsorted(CLASSES, official.model); prompt = official.prompt_id.values; n = len(y)
    with np.load(a.collect, allow_pickle=False) as z:
        topic, pp, pc, biases = z['topic'].astype(np.float64), z['plain'], z['centered'], z['bias']
        ex_group, ex_y, ids = z['external_groups'], z['external_y'], z['ids'].astype(str)
        if int(z['official_rows']) != n:
            raise ValueError('Official/external boundary changed')
    if rec['official_sha256'] != sha(BASE / 'data/original/train.csv') or not np.array_equal(ids[:n], official.id.astype(str)):
        raise ValueError('Collector row identity changed')
    from v51_collect import NATIVE
    states = [json.loads((NATIVE / f'calibration/fold{k}.json').read_text())['models']['qwen8'] for k in range(5)]
    # Use the pinned native OOF, not its scores on rows used in that fold's training.
    import shared  # noqa: F401
    import common as C
    import sage as S
    public_pp, public_pc, public_fold = C.public_native_oof()
    if not np.array_equal(public_fold, fold):
        raise ValueError('Native split assignment changed')
    # Verify inference parity; accepting new scores with a different row order is forbidden.
    rows = np.arange(n)
    parity = max(float(np.max(np.abs(pp[fold, rows] - public_pp))), float(np.max(np.abs(pc[fold, rows] - public_pc))))
    if parity > .05:
        raise ValueError(f'Frozen native OOF parity failed: {parity}')
    pp[:, :n] = np.broadcast_to(public_pp, (5, n, 18))
    pc[:, :n] = np.broadcast_to(public_pc, (5, n, 18))
    models, taus, selections = [], [], []
    for k in range(5):
        # Remove external panel class-composition bias using EXTERNAL labels only.
        external = pc[k, n:]
        class_means = np.stack([external[ex_y == c].mean(0) for c in range(18)])
        adjusted = external - class_means[ex_y] + class_means.mean(0)
        model, tau, selected = EB.select_external(topic[n:], adjusted, ex_group)
        models.append(model); taus.append(tau); selections.append(selected)
    baseline, candidate = np.zeros((n, 18)), np.zeros((n, 18))
    correct0, correct1 = np.zeros(n), np.zeros(n)
    by_draw = []
    for draw in range(10):
        grouping = sparse_groups(prompt, 2000 + draw)
        z0 = EB.calibrate(pp[:, :n], pc[:, :n], biases, topic[:n], grouping, states)
        z1 = EB.calibrate(pp[:, :n], pc[:, :n], biases, topic[:n], grouping, states, models, taus)
        p0, p1 = np.exp(z0[fold, rows]), np.exp(z1[fold, rows])
        baseline += p0 / 10; candidate += p1 / 10
        c0, c1 = p0.argmax(1) == y, p1.argmax(1) == y
        correct0 += c0 / 10; correct1 += c1 / 10
        by_draw.append(dict(seed=2000 + draw, base_ba=100 * ba(y, c0), eb_ba=100 * ba(y, c1)))
    eb_report = paired_ci(y, prompt, correct0, correct1, draws=a.bootstrap)
    # SAGE uses ONE fixed sparse grouping, exactly one input dataset, as at deployment.
    # Averaging alternate partitions would give it information unavailable on hidden test.
    g = sparse_groups(prompt, 2000)
    b0 = EB.calibrate(pp[:, :n], pc[:, :n], biases, topic[:n], g, states)
    b1 = EB.calibrate(pp[:, :n], pc[:, :n], biases, topic[:n], g, states, models, taus)
    probs = np.exp((b1 if eb_report['ci95'][0] > 0 else b0)[fold, rows])
    artifacts = {str(a.collect.resolve()): sha(a.collect), str(a.collect.with_suffix('.json').resolve()): sha(a.collect.with_suffix('.json'))}
    group_scores, reports, all_oof = {}, {}, np.full((n, 18), np.nan)
    for group in ([] if a.eb_only else GROUPS):
        scores = {}
        for excluded in [*((k,) for k in range(5)), *itertools.combinations(range(5), 2)]:
            scores[excluded] = expert_scores(a.experts, group, excluded, official, fold, artifacts)
            info = json.loads((a.experts / group / split_name(excluded) / 'scores.json').read_text())
            if info['identity']['external_sha256'] != rec['identity']['external']:
                raise ValueError('Experts and EB collector use different external data')
        group_scores[group] = scores
        columns = [CLASSES.index(c) for c in GROUPS[group]]
        for k in range(5):
            r = np.flatnonzero(fold == k); all_oof[np.ix_(r, columns)] = scores[(k,)][np.ix_(r, columns)]
        use = np.isin(y, columns)
        expert_pred = np.asarray(columns)[all_oof[use][:, columns].argmax(1)]
        native_pred = np.asarray(columns)[probs[use][:, columns].argmax(1)]
        pred, w = nested_sage(probs, y, fold, group_scores, [group], S)
        family = paired_ci(y, prompt, probs.argmax(1), pred, metric='conditional', draws=a.bootstrap)
        reports[group] = dict(within_native=100 * float((native_pred == y[use]).mean()),
                             within_expert=100 * float((expert_pred == y[use]).mean()),
                             within_delta=100 * float((expert_pred == y[use]).mean() - (native_pred == y[use]).mean()),
                             family=family, outer_weights=w)
    passing = [g for g in reports if reports[g]['within_delta'] > 0 and reports[g]['family']['ci95'][0] > 0]
    combined = None; pred = probs.argmax(1)
    if passing:
        pred, _ = nested_sage(probs, y, fold, group_scores, passing, S)
        combined = paired_ci(y, prompt, probs.argmax(1), pred, metric='conditional', draws=a.bootstrap)
    enabled = choose_components(eb_report, reports, combined)
    a.out.mkdir(parents=True, exist_ok=True)
    save_npz(a.out / 'eb_models.npz', **EB.pack(models, taus))
    weights = B.fit(probs, all_oof, enabled['experts'], y, S) if enabled['experts'] else []
    dump(a.out / 'sage_bundle.json', dict(weights=list(weights), enabled=enabled['experts'], classes=CLASSES,
         method='v51_sage_discriminative', C=.3, sage_sha256=sha(S.__file__)))
    save_npz(a.out / 'oof_predictions.npz', y=y, fold=fold, prompt=prompt.astype(str), base=baseline,
             eb=candidate, sage_pred=pred, expert_lp=all_oof)
    for name in ('eb_models.npz', 'sage_bundle.json', 'oof_predictions.npz'):
        path = a.out / name; artifacts[str(path.resolve())] = sha(path)
    for name in ('v51_eb.py', 'v51_expert.py', 'v51_common.py', 'v51_sage.py', 'v51_evaluate.py'):
        path = ROOT / 'scripts' / name; artifacts[str(path.resolve())] = sha(path)
    dump(a.out / 'decision.json', dict(method='v51_eb_sibling_experts', enabled=enabled, eb=eb_report,
         groups=reports, combined=combined, eb_external_selection=selections, grouping_draws=by_draw,
         native_parity_max_abs=parity, base_family_conditional=100 * conditional(y, probs.argmax(1)),
         base_family_accuracy=100 * float((FAMILY[y] == FAMILY[probs.argmax(1)]).mean()),
         metric_scope='official_train_prompt_OOF_validation_native_alone', hidden_test_evaluated=False,
         proxy_used_for_acceptance=False, artifacts=artifacts,
         negative_result_action='do_not_build_or_submit; new API data requires a separate user budget decision'))
    print((a.out / 'decision.json').read_text())


if __name__ == '__main__':
    main()
