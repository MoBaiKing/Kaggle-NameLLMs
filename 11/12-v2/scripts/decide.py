"""Validation proxy for enumerated GRPO plus multi-model voting; deploy declared folds."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import ROOT
from selection import assess
from public_parent import load_parent
from windows import sha
import common as C

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs', type=Path, default=ROOT / 'runs_grpo')
p.add_argument('--folds', default='0,1,2,3,4')
p.add_argument('--out', type=Path, default=ROOT / 'runs_grpo/decision.json')
a = p.parse_args()
folds = [int(k) for k in a.folds.split(',')]
if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
    p.error('Predeclare distinct public fold IDs 0..4')
original_parent = C.public_lora_stack()
vote_parent, parent_record = load_parent(C.pid, C.classes)
pp, pc, pfold = C.public_native_oof()
rows = np.concatenate([C.FOLDS[k] for k in folds])
original, same_vote, optimized = [np.zeros((len(rows), len(C.classes))) for _ in range(3)]
pos = {int(r): i for i, r in enumerate(rows)}
per_fold, fingerprints = {}, {}
for k in folds:
    root = a.runs / f'fold{k}'
    done = json.loads((root / 'done.json').read_text())
    check = json.loads((root / 'native_eval.json').read_text())
    if not check.get('parity_ok') or not check.get('artifact_sha256'):
        raise ValueError(f'Fold {k} has not passed native export/runtime parity')
    if (done.get('method') != 'public_native_grpo_v2' or done.get('optimizer_steps', 0) < 1
            or done.get('calibration_mode') != 'frozen_public'):
        raise ValueError(f'Fold {k} is not a completed v2 public-native continuation')
    if done['voting']['parent_record'] != parent_record:
        raise ValueError('Frozen public voter identities differ from training')
    for record in (done, check):
        for filename, expected in record['artifact_sha256'].items():
            if sha(root / filename) != expected:
                raise ValueError(f'Changed evaluated artifact: fold{k}/{filename}')
    if sha(root / 'trainables.safetensors') == done['public_sha256']['state']:
        raise ValueError(f'Fold {k} exported unchanged public weights')
    fingerprints[str(k)] = dict(done=sha(root / 'done.json'), evaluation=sha(root / 'native_eval.json'),
                                state=sha(root / 'trainables.safetensors'),
                                calibration=sha(root / 'calibration/fold.json'))
    with np.load(root / 'native_eval.npz', allow_pickle=False) as z:
        val = z['val']; assert set(val) == set(C.FOLDS[k])
        ours = json.loads((root / 'calibration/fold.json').read_text())['models']['qwen8']
        public = C.public_state(k)
        if ours != public:
            raise ValueError('Public calibration was not frozen')
        ix = np.array([pos[int(r)] for r in val])
        for draw in range(10):
            group = C.groups_sized(C.pid[val], 2, 2, 1000 + draw)
            lo = C.state_lp(z['plain_ours'], z['centered_ours'], z['bias_ours'], ours, group)
            lp = C.state_lp(pp[val], pc[val], C.public_bias(k), public, group)
            original[ix] += np.exp(C.lsm(.60 * original_parent[val] + .40 * lp)) / 10
            same_vote[ix] += (.60 * np.exp(vote_parent[val]) + .40 * np.exp(lp)) / 10
            optimized[ix] += (.60 * np.exp(vote_parent[val]) + .40 * np.exp(lo)) / 10
        per_fold[str(k)] = dict(optimized_vote_ba=100 * C.ba(C.y_all[val], optimized[ix].argmax(1)),
                               public_same_vote_ba=100 * C.ba(C.y_all[val], same_vote[ix].argmax(1)),
                               original_public_proxy_ba=100 * C.ba(C.y_all[val], original[ix].argmax(1)))
# Even untouched native slots use the new voting rule. Keep that change separate
# from the trained-state contribution when only a subset of folds is optimized.
fraction = len(folds) / 5
proxy_deployment = (1 - fraction) * same_vote + fraction * optimized
reports = {'overall_vs_original': assess(C.y_all[rows], C.pid[rows], original, proxy_deployment, 1.),
           'grpo_vs_same_vote': assess(C.y_all[rows], C.pid[rows], same_vote, proxy_deployment, 1.)}
for report in reports.values():
    report['positive_uplift_evidence'] = report.pop('passed')
out = dict(decision='grpo', method='public_native_grpo_v2', our_folds_used=folds,
           rule='Deploy all predeclared optimized folds and frozen voting weights',
           split='official train OOF VALIDATION proxy, not hidden test or complete Kaggle stack',
           per_fold=per_fold, candidates=reports, artifacts=fingerprints,
           parent_record=parent_record, hidden_test_evaluated=False)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
