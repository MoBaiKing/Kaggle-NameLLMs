"""Real four-rank trainer: public-state loading, mid-rollout resume and export.

Only synthetic data and a tiny random 36-layer model run on CPU. Production
public-asset validation is exercised against a fully pinned synthetic manifest.
"""
import datetime as dt
import importlib.util
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / '12/tests'))
import grpo_train as G
import public_parent as P
import train as S
from test_pipeline import model_factory
from windows import sha


def fixture(root):
    root = Path(root)
    for directory in ['data', 'model', 'native/fold0', 'native/calibration', 'meta/fold0', 'config', 'data/reference']:
        (root / directory).mkdir(parents=True)
    frame = pd.DataFrame([dict(text=f'synthetic answer {prompt} {label}', prompt_id=f'p{prompt}',
                               model=f'class{label:02}') for prompt in range(6) for label in range(18)])
    frame.to_csv(root / 'data/train.csv', index=False)
    fit = np.flatnonzero(frame.prompt_id.values != 'p5').tolist()
    val = np.flatnonzero(frame.prompt_id.values == 'p5').tolist()
    (root / 'folds.json').write_text(json.dumps({'0': val}))
    split = dict(training_rows=fit, validation_rows=val, official_oof_eligible=True)
    (root / 'meta/fold0/split.json').write_text(json.dumps(split))
    calibration = dict(plain_scale=1.1, centered_scale=.9, selected_centered_weight=.35,
                       outer_labels_used=False, centered_head_trained=True)
    (root / 'native/calibration/fold0.json').write_text(json.dumps({'models': {'qwen8': calibration}}))
    for name in ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'merges.txt', 'vocab.json']:
        (root / 'model' / name).write_text('{}')
    torch.manual_seed(9)
    args = SimpleNamespace(lora_drop=0., feat_drop=0., proj_dim=256)
    _, model = model_factory(args, 18, torch.device('cpu'))
    # Native export/reload must preserve both heads and every adapter exactly.
    S.export_public(model, root / 'native/fold0/trainables.safetensors')
    expected = {k: v.clone() for k, v in S.trainable_state(model).items() if not k.startswith('proj.')}
    torch.manual_seed(100)
    _, target = model_factory(args, 18, torch.device('cpu'))
    G.load_public(target, root / 'native/fold0/trainables.safetensors')
    state = S.trainable_state(target)
    assert all(torch.equal(state[k], v) for k, v in expected.items())
    hashes = {f'base/{p.name}': sha(p) for p in (root / 'model').iterdir()}
    hashes.update({'fold0/trainables.safetensors': sha(root / 'native/fold0/trainables.safetensors'),
                   'calibration/fold0.json': sha(root / 'native/calibration/fold0.json'),
                   'fold0/split.json': sha(root / 'meta/fold0/split.json')})
    manifest = dict(classes=sorted(frame.model.unique()), revision='synthetic-revision', file_hashes=hashes)
    (root / 'native/manifest.json').write_text(json.dumps(manifest))
    spec = dict(native_assets=dict(manifest_sha256=sha(root / 'native/manifest.json')),
                model=dict(revision='synthetic-revision'),
                original_data={'train.csv': dict(sha256=sha(root / 'data/train.csv'))},
                folds_sha256=sha(root / 'folds.json'))
    references = {}
    rng = np.random.default_rng(91)
    for name in P.WEIGHTS:
        path = root / 'data/reference' / f'{name}.npz'
        np.savez(path, labels=np.array(sorted(frame.model.unique())),
                 lp=rng.normal(size=(5, len(frame), 18)).astype(np.float32))
        references[path.name] = dict(sha256=sha(path))
    spec['reference_data'] = references
    (root / 'config/assets.json').write_text(json.dumps(spec))


def worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    root = Path(directory); G.BASE = root; P.BASE = root
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    common = ['--data', str(root / 'data'), '--model', str(root / 'model'),
              '--folds', str(root / 'folds.json'), '--native-dir', str(root / 'native'),
              '--native-meta', str(root / 'meta'), '--token-cache', str(root / 'tokens'),
              '--fold', '0', '--deadline', future, '--device', 'cpu', '--max-batches', '1',
              '--micro-batch', '1', '--updates', '2', '--lr', '.0001']
    # Verify malformed inputs are rejected using the production validator.
    args = G.parse(common)
    frame = pd.read_csv(root / 'data/train.csv', dtype=str)
    fingerprint = G.token_identity(args)
    G.verify_public(args, frame, fingerprint)
    bad = dict(fingerprint, train='changed')
    try:
        G.verify_public(args, frame, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed official data was accepted')
    # Produce a genuine v1 reference with the original frozen engine, then v2
    # must reuse its bytes while initializing from PUBLIC, not the v1 student.
    spec = importlib.util.spec_from_file_location('frozen_v1_engine', G.V1 / 'scripts/grpo_train.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    legacy.BASE = root
    legacy.main(common + ['--out', str(root / 'v1_source')], model_factory=model_factory)
    common += ['--reference-source', str(root / 'v1_source')]
    original_step = torch.optim.AdamW.step
    def pause_after_step(opt, *a, **kw):
        value = original_step(opt, *a, **kw)
        os.kill(os.getpid(), signal.SIGTERM)
        return value
    torch.optim.AdamW.step = pause_after_step
    paused_args = common + ['--out', str(root / 'resumed')]
    try:
        G.main(paused_args, model_factory=model_factory)
    except SystemExit as error:
        assert error.code == 75
    else:
        raise AssertionError('The interrupted rollout must pause')
    finally:
        torch.optim.AdamW.step = original_step
    checkpoint = root / 'resumed/fold0/checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['state']['update'] == 1 and state['state']['pending'] is not None
    assert state['world_size'] == 4 and len(state['rng_states']) == 4
    assert not checkpoint.with_name('done.json').exists()
    G.main(paused_args, model_factory=model_factory)
    done = json.loads(checkpoint.with_name('done.json').read_text())
    assert done['optimizer_steps'] == 2 and done['resumes'] == 1
    assert done['calibration_mode'] == 'frozen_public' and done['world_size'] == 4
    assert done['method'] == 'public_native_grpo_v2' and done['estimator'] == 'enumerated_full_support'
    reference = root / 'resumed/fold0/reference.npz'
    assert reference.is_symlink()
    assert reference.resolve() == (root / 'v1_source/fold0/reference.npz').resolve()
    assert sha(reference) == sha(root / 'v1_source/fold0/reference.npz')
    assert len(np.load(checkpoint.with_name('inner_rows.npy'))) == 0
    G.main(common + ['--out', str(root / 'uninterrupted')], model_factory=model_factory)
    resumed = load_file(str(root / 'resumed/fold0/trainables.safetensors'))
    uninterrupted = load_file(str(root / 'uninterrupted/fold0/trainables.safetensors'))
    assert all(torch.equal(v, uninterrupted[k]) for k, v in resumed.items())
    source = load_file(str(root / 'native/fold0/trainables.safetensors'))
    assert any(not torch.equal(v, source[k]) for k, v in resumed.items())
    # Completed reuse bypasses model loading, but changed protocols still fail.
    G.main(paused_args, model_factory=lambda *a: (_ for _ in ()).throw(AssertionError('Retraining completed fold')))
    try:
        G.main(paused_args + ['--beta', '.03'], model_factory=model_factory)
    except ValueError as error:
        assert 'Completed GRPO' in str(error)
    else:
        raise AssertionError('Changed objectives reused a completed run')
    if rank == 0:
        print(json.dumps(dict(world_size=4, public_native_loaded=True, split_guard=True,
                              mid_rollout_resume_exact=True, tensors=len(resumed),
                              weights_updated=True, completed_reused=True)), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='grpo_v2_pipeline_') as directory:
        fixture(directory)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, directory), nprocs=4, join=True)
