"""Real v3 trainer on four Gloo ranks (CPU, tiny random 36-layer model, synthetic data): public-state loading,
pause/resume exactness, export + WiSE, completed-run reuse, protocol guard, and 4-rank == 1-rank updates.
No pretrained asset or GPU; this is a code test, not an 8B run."""
import datetime as dt
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
import train_v3 as G
import oof_targets as T
sys.path.insert(1, str(HERE.parents[2] / '12-v2/tests'))
sys.path.insert(2, str(HERE.parents[2] / '12/tests'))
from test_pipeline_v2 import fixture
from test_pipeline import model_factory
from windows import sha


def targets(root):
    frame = pd.read_csv(root / 'data/train.csv', dtype=str)
    rng = np.random.default_rng(5)
    parent = rng.normal(size=(len(frame), 18)); parent -= np.log(np.exp(parent).sum(1, keepdims=True))
    p_true = rng.uniform(size=len(frame))
    T.write(root / 'targets.npz', dict(parent_lp=parent.astype(np.float32), p_true=p_true, weight=T.weight_of(p_true),
                                       row_fold=np.zeros(len(frame), int)),
            dict(method='synthetic', eps=T.EPS, gamma=T.GAMMA, parent_temperature=1.75,
                 train_sha256=sha(root / 'data/train.csv'), folds_sha256=sha(root / 'folds.json')))


def arguments(root):
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    return ['--data', str(root / 'data'), '--model', str(root / 'model'), '--folds', str(root / 'folds.json'),
            '--native-dir', str(root / 'native'), '--native-meta', str(root / 'meta'),
            '--token-cache', str(root / 'tokens'), '--targets', str(root / 'targets.npz'),
            '--fold', '0', '--deadline', future, '--device', 'cpu', '--max-batches', '3',
            '--micro-batch', '1', '--lr-lora', '1e-3', '--lr-head', '1e-2']


def worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    root = Path(directory); G.V2T.BASE = root
    common = arguments(root)
    original_step = torch.optim.AdamW.step
    def pause_after_step(opt, *a, **kw):
        value = original_step(opt, *a, **kw)
        os.kill(os.getpid(), signal.SIGTERM)
        return value
    torch.optim.AdamW.step = pause_after_step
    paused = common + ['--out', str(root / 'resumed')]
    try:
        G.main(paused, model_factory=model_factory)
    except SystemExit as error:
        assert error.code == 75
    else:
        raise AssertionError('Interrupted training must pause')
    finally:
        torch.optim.AdamW.step = original_step
    checkpoint = root / 'resumed/fold0/checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['state']['batch'] == 1 and state['state']['optimizer_steps'] == 1
    assert state['world_size'] == 4 and len(state['rng_states']) == 4
    assert not checkpoint.with_name('done.json').exists()
    G.main(paused, model_factory=model_factory)
    done = json.loads(checkpoint.with_name('done.json').read_text())
    assert done['optimizer_steps'] == 3 and done['resumes'] == 1 and done['method'] == 'public_native_oofce_v3'
    assert done['calibration_mode'] == 'frozen_public' and done['world_size'] == 4 and done['wise_alpha'] == .5
    G.main(common + ['--out', str(root / 'uninterrupted')], model_factory=model_factory)
    resumed = load_file(str(root / 'resumed/fold0/trainables.safetensors'))
    straight = load_file(str(root / 'uninterrupted/fold0/trainables.safetensors'))
    assert all(torch.equal(v, straight[k]) for k, v in resumed.items())
    public = load_file(str(root / 'native/fold0/trainables.safetensors'))
    assert any(not torch.equal(v, public[k]) for k, v in resumed.items())
    wise = load_file(str(root / 'resumed/fold0/trainables_wise050.safetensors'))
    assert all(torch.allclose(wise[k], ((public[k].double() + resumed[k].double()) / 2).float()) for k in public)
    G.main(paused, model_factory=lambda *a: (_ for _ in ()).throw(AssertionError('Retrained a completed fold')))
    try:
        G.main(paused + ['--beta', '.1'], model_factory=model_factory)
    except ValueError as error:
        assert 'Completed v3' in str(error)
    else:
        raise AssertionError('A changed objective reused a completed run')


def single(directory):
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(name, None)
    root = Path(directory); G.V2T.BASE = root
    G.main(arguments(root) + ['--out', str(root / 'single')], model_factory=model_factory)
    one = load_file(str(root / 'single/fold0/trainables.safetensors'))
    four = load_file(str(root / 'uninterrupted/fold0/trainables.safetensors'))
    public = load_file(str(root / 'native/fold0/trainables.safetensors'))
    gap = max(float((one[k] - four[k]).abs().max()) for k in one)
    moved = max(float((four[k] - public[k]).abs().max()) for k in four)
    assert gap < 1e-4 * max(1., moved) and moved > 1e-4, (gap, moved)
    return gap, moved, len(one)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v3_pipeline_') as directory:
        fixture(directory); targets(Path(directory))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, directory), nprocs=4, join=True)
        gap, moved, tensors = single(directory)
        print(json.dumps(dict(world_size=4, public_native_loaded=True, pause_resume_exact=True, wise_exact=True,
                              completed_reused=True, protocol_guard=True, four_vs_one_rank_max_abs_gap=gap,
                              max_weight_change=moved, tensors=tensors)))
