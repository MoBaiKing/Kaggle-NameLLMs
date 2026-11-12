"""CPU-only trainer pause/resume/export and Notebook construction checks."""
import copy
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_submission as B
import train as S


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    pad_token_id = 0
    padding_side = 'right'

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            assert texts == '' and add_special_tokens
            return {'input_ids': [S.TERMINAL]}
        return {'input_ids': [[1, 2 + len(str(t)) % 97, 3] for t in texts]}

    def pad(self, rows, **kwargs):
        width = max(map(len, rows['input_ids']))
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows['input_ids']])
        return Batch(input_ids=ids, attention_mask=(ids != 0).long())


def model_factory(args, classes, device):
    config = Qwen3Config(vocab_size=151650, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=36, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                         attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    return S.assemble(Qwen3Model(config), Tokenizer(), args, classes, device)


def trainer_test(temp):
    # All real training code and the 36-layer export layout are exercised, with
    # tiny random weights and synthetic external text; no pretrained asset/GPU.
    classes = sorted(pd.read_csv(ROOT / 'data/original/train.csv').model.unique())
    stream = pd.DataFrame([dict(text=f'external fixture {c} {j}', model=c,
                                prompt=f'fixture{j}', w=1.)
                           for c in classes for j in range(3)])
    stream.to_parquet(temp / 'pool.parquet', index=False)
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    args = ['--data', str(ROOT / 'data/original'), '--model', 'unused',
            '--folds', str(ROOT / 'data/folds.json'), '--fold', '0',
            '--out', str(temp / 'runs'), '--stream', str(temp / 'pool.parquet'),
            '--train-end', future, '--stage2-end', past, '--device', 'cpu',
            '--max-steps', '1', '--max-steps2', '1', '--micro-batch', '4',
            '--proj-dim', '8', '--seed', '42']
    try:
        S.main(args, model_factory=model_factory)
    except SystemExit as error:
        assert error.code == 75, error
    else:
        raise AssertionError('Incomplete stage 2 must pause')
    run = temp / 'runs/fold0'
    assert (run / 'checkpoint.pt').is_file()
    assert not (run / 'done.json').exists()
    assert not (run / 'trainables.safetensors').exists()
    state = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['world_size'] == 1 and state['st']['stage'] == 2
    assert state['st']['step'] == 1 and state['st']['step2'] == 0
    assert len(state['rng_states']) == 1
    args[args.index('--stage2-end') + 1] = future
    S.main(args, model_factory=model_factory)
    done = json.loads((run / 'done.json').read_text())
    assert done['resumes'] == 1 and done['steps_stage1'] == done['steps_stage2'] == 1
    assert not (run / 'checkpoint.pt').exists()
    with safe_open(run / 'trainables.safetensors', framework='pt') as weights:
        assert len(weights.keys()) == 524
        assert all(torch.isfinite(weights.get_tensor(k)).all() for k in weights.keys())
    probs = np.load(run / 'val_probs_final.npy')
    assert probs.shape == (len(json.loads((ROOT / 'data/folds.json').read_text())['0']), 18)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1., atol=1e-5)
    print(json.dumps(dict(trainer_pause_resume_export=True, exported_tensors=524,
                          validation_rows=len(probs))), flush=True)


def notebook_test(temp):
    import glob
    import shutil
    calibration = {'models': {'qwen8': dict(outer_labels_used=False,
                    centered_head_trained=True, plain_scale=1., centered_scale=1.,
                    selected_centered_weight=.25)}}
    public = temp / 'public'
    (public / 'base').mkdir(parents=True)
    (public / 'base/model.safetensors.index.json').write_text('{}')
    (public / 'manifest.json').write_text('{}')
    (public / 'calibration').mkdir()
    runs = temp / 'notebook_runs'
    for k in range(5):
        (public / f'fold{k}').mkdir()
        save_file({'plain.weight': torch.ones(18, 4)}, public / f'fold{k}/trainables.safetensors')
        (public / f'calibration/fold{k}.json').write_text(json.dumps(calibration))
        (runs / f'fold{k}/calibration').mkdir(parents=True)
        save_file({'plain.weight': torch.full((18, 4), 2.)}, runs / f'fold{k}/trainables.safetensors')
        (runs / f'fold{k}/calibration/fold.json').write_text(json.dumps(calibration))
    # A local pinned upstream Notebook also validates the real injection anchor.
    # Without external downloads, a minimal fixture keeps this test standalone.
    original_src, original_sha = B.SRC, B.SRC_SHA
    if not (B.SRC / 'ntl-684-ozertuu-mirror.ipynb').exists():
        B.SRC = temp / 'notebook_fixture'; B.SRC.mkdir()
        fixture = dict(cells=[dict(cell_type='code', metadata={}, outputs=[],
                                  execution_count=None, source=[s])
                              for s in ('pass', 'pass', B.ANCHOR)], metadata={},
                       nbformat=4, nbformat_minor=5)
        path = B.SRC / 'ntl-684-ozertuu-mirror.ipynb'
        path.write_text(json.dumps(fixture))
        B.SRC_SHA = hashlib.sha256(path.read_bytes()).hexdigest()
        (B.SRC / 'kernel-metadata.json').write_text(json.dumps(dict(
            id='upstream/fixture', dataset_sources=[], enable_internet=False)))
    try:
        upstream = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
        for decision, used in [('ours2', [0, 1]), ('ours4', [0, 1, 2, 3])]:
            d = temp / f'{decision}.json'
            d.write_text(json.dumps(dict(decision=decision, our_folds_used=used)))
            output = temp / decision
            B.build(d, runs, output, 'reproducer')
            notebook = json.loads((output / 'final/submission.ipynb').read_text())
            metadata = json.loads((output / 'final/kernel-metadata.json').read_text())
            assert metadata['id'] == 'reproducer/ntl-11-11'
            assert metadata['is_private'] and not metadata['enable_internet']
            expected = copy.deepcopy(upstream)
            if used:
                src = ''.join(expected['cells'][2]['source'])
                expected['cells'][2]['source'] = (src.replace(B.ANCHOR, B.ANCHOR + B.PATCH)).splitlines(keepends=True)
            assert notebook['cells'][1:] == expected['cells']
            if used:
                mix = temp / f'{decision}_mix'
                patch = B.PATCH.replace('/kaggle/input/**/dyl_11_11_native.json',
                                        str(output / 'upload/dyl_11_11_native.json'))
                patch = patch.replace('/kaggle/temp/dyl_native_mix', str(mix))
                exec(compile(patch, '<native-slot-patch>', 'exec'), dict(
                    NATIVE=str(public), glob=glob, os=os, json=json, shutil=shutil))
                for slot in range(5):
                    path = mix / f'fold{slot}/trainables.safetensors'
                    with safe_open(path, framework='pt') as state:
                        value = state.get_tensor('plain.weight')[0, 0].item()
                    assert value == (2. if slot < len(used) else 1.)
        print(json.dumps(dict(notebook_build_and_native_slots=True,
                              decisions=['ours2', 'ours4'])), flush=True)
    finally:
        B.SRC, B.SRC_SHA = original_src, original_sha


if __name__ == '__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='name_llm_pipeline_') as directory:
        trainer_test(Path(directory))
        notebook_test(Path(directory))
