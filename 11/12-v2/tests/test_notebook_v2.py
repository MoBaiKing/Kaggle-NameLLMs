"""Offline Notebook injection executes soft voting and rejects incomplete members."""
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_submission as B
from runtime_vote import multi_model_parent
from windows import sha

ENGINE = '''%%writefile infer.py
import numpy as np
def native_stage(P):
    folds = z['folds'].tolist()
    share = .40
    logP = np.log(P)
    probs = np.zeros_like(P)
    for i in range(len(folds)):
        probs += rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i]))) / len(folds)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)
    return probs
if False:
    if False:
        log('NATIVE SKIPPED: not enough time left')
    else:
        try:
            pass
        except Exception as _e:
            log('NATIVE SKIPPED:', repr(_e))
'''


def test(root):
    old_src, old_sha = B.SRC, B.SRC_SHA
    public, runs = root / 'public', root / 'runs'
    for directory in ['public/base', 'public/calibration', 'source']:
        (root / directory).mkdir(parents=True)
    (public / 'manifest.json').write_text('{}')
    B.SRC = root / 'source'
    notebook = dict(cells=[dict(cell_type='code', metadata={}, source=[source], outputs=[], execution_count=None)
                          for source in ['pass', 'pass', B.ANCHOR, ENGINE]],
                    metadata={}, nbformat=4, nbformat_minor=5)
    source = B.SRC / 'ntl-684-ozertuu-mirror.ipynb'; source.write_text(json.dumps(notebook))
    B.SRC_SHA = sha(source)
    (B.SRC / 'kernel-metadata.json').write_text(json.dumps(dict(dataset_sources=[], enable_internet=False)))
    cal = {'models': {'qwen8': dict(plain_scale=1., centered_scale=1., selected_centered_weight=.35,
                                  outer_labels_used=False, centered_head_trained=True)}}
    artifacts = {}
    for k in range(5):
        (public / f'fold{k}').mkdir()
        save_file({'plain.weight': torch.ones(18, 4)}, public / f'fold{k}/trainables.safetensors')
        (public / f'calibration/fold{k}.json').write_text(json.dumps(cal))
        run = runs / f'fold{k}'; (run / 'calibration').mkdir(parents=True)
        save_file({'plain.weight': torch.full((18, 4), 2.)}, run / 'trainables.safetensors')
        (run / 'calibration/fold.json').write_text(json.dumps(cal))
        (run / 'done.json').write_text(json.dumps(dict(method='public_native_grpo_v2',
            estimator='enumerated_full_support', optimizer_steps=2, public_sha256={'state': 'different source'},
            protocol={'code': {'runtime_vote.py': sha(Path(B.__file__).with_name('runtime_vote.py'))}})))
        (run / 'native_eval.json').write_text(json.dumps({'parity_ok': True}))
        artifacts[str(k)] = {key: sha(run / name) for key, name in
                            [('done', 'done.json'), ('evaluation', 'native_eval.json'),
                             ('state', 'trainables.safetensors'), ('calibration', 'calibration/fold.json')]}
    try:
        for case, folds in enumerate([list(range(5)), [1, 3]]):
            decision = root / f'plan{case}.json'
            decision.write_text(json.dumps(dict(decision='grpo', method='public_native_grpo_v2',
                                                 our_folds_used=folds, artifacts=artifacts)))
            out = root / f'build{case}'; B.build(decision, runs, out, 'reproducer')
            metadata = json.loads((out / 'final/kernel-metadata.json').read_text())
            assert metadata['id'] == 'reproducer/ntl-11-12-grpo-v2'
            spec = json.loads((out / 'upload/dyl_11_12_v2_native.json').read_text())
            assert sorted(v[1] for v in spec['slots'].values()) == list(range(5))
            patch = B.PATCH.replace('/kaggle/input/**/dyl_11_12_v2_native.json',
                                    str(out / 'upload/dyl_11_12_v2_native.json'))
            patch = patch.replace('/kaggle/temp/dyl_native_v2_mix', str(root / f'mix{case}'))
            exec(compile(patch, '<v2-native-load>', 'exec'), dict(
                 NATIVE=str(public), glob=glob, os=os, json=json, shutil=shutil))
            nb = json.loads((out / 'final/submission.ipynb').read_text())
            source = next(''.join(c['source']) for c in nb['cells']
                          if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n'))
            assert 'NATIVE SKIPPED' not in source
            softmax = lambda x: np.exp(x - np.logaddexp.reduce(x, -1, keepdims=True))
            public_p = np.array([[.4, .4, .2]])
            terms = {name: weight * np.log([[.2, .3, .5]]) for name, weight in
                     [('q14v5', 3.), ('q8v5', 2.), ('ll8v5', 1.)]}
            native_p = np.array([[.1, .2, .7]])
            scope = dict(z={'folds': np.arange(5), 'finite': np.ones(1, bool)}, code=0,
                         LP=terms, lp=np.repeat(np.log(native_p)[None], 5, axis=0),
                         rowsoftmax=softmax, decide_scores=np.log, log=lambda *x: None)
            exec(compile(source.partition('\n')[2], '<v2-offline-engine>', 'exec'), scope)
            output = scope['native_stage'](public_p)
            expected = .6 * multi_model_parent(public_p, terms) + .4 * native_p
            assert np.allclose(output, expected)
            scope['z']['folds'] = np.arange(4)
            try:
                scope['native_stage'](public_p)
            except AssertionError:
                pass
            else:
                raise AssertionError('Incomplete native inference was accepted')
            scope['z']['folds'] = np.arange(5); scope['LP'] = {'q8v5': terms['q8v5']}
            try:
                scope['native_stage'](public_p)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Missing strong voters were accepted')
        print(json.dumps(dict(v2_notebook=True, actual_soft_vote_executed=True,
                              five_folds_required=True, three_strong_voters_required=True)))
    finally:
        B.SRC, B.SRC_SHA = old_src, old_sha


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v2_notebook_') as directory:
        test(Path(directory))
