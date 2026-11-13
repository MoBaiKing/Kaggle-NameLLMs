"""A1 trainer on four Gloo ranks (CPU, tiny random 36-layer model, synthetic official + external rows): official and
sibling-structured replay cells alternate, losses are finite, pause/resume == uninterrupted, WiSE export, reuse,
protocol guard. Loss units: hierarchical SupCon grading, sibling margin, replay cells. Code test, not an 8B run."""
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
import a1_train as G
import a1_objective as A
import families as F
import oof_targets as T
sys.path.insert(1, str(HERE.parents[2] / '12-v2/tests'))
sys.path.insert(2, str(HERE.parents[2] / '12/tests'))
from test_pipeline_v2 import fixture
from test_pipeline import model_factory
from windows import sha


def unit():
    torch.manual_seed(0)
    labels = torch.tensor([F.CLASSES.index(c) for c in ['gemini-3.7-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'kimi-k3']])
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    base = A.hierarchical_supcon(z, labels)
    # graded positives: pulling a sibling (gemini-3.8) next to the gemini-3.7 pair costs less than pulling an
    # unrelated class (kimi) there, and pulling the unrelated class there is worse than leaving it
    z2 = z.clone(); z2[2] = torch.nn.functional.normalize(z[0] + z[1], dim=-1)
    z3 = z.clone(); z3[3] = torch.nn.functional.normalize(z[0] + z[1], dim=-1)
    assert A.hierarchical_supcon(z3, labels) > base and A.hierarchical_supcon(z2, labels) < A.hierarchical_supcon(z3, labels)
    lp = torch.full((1, 18), -10.); y = torch.tensor([F.CLASSES.index('glm-5.3')])
    lp[0, y] = -.1; lp[0, F.CLASSES.index('glm-5.3-flash')] = -3.
    good = A.sibling_margin(lp.log_softmax(-1), y)
    lp[0, F.CLASSES.index('glm-5.3-flash')] = -.05
    assert A.sibling_margin(lp.log_softmax(-1), y) > good
    assert float(A.sibling_margin(torch.zeros(1, 18).log_softmax(-1), torch.tensor([F.CLASSES.index('qwen3.8-max')]))) == 0
    rng = np.random.default_rng(0)
    labels_np = np.repeat(np.arange(18), 30); prompts = np.array([f'p{i % 40}' for i in range(len(labels_np))])
    cells = A.pool_cells(labels_np, prompts, np.arange(len(labels_np)), 50, 16, rng)
    assert all(len(c) == 16 for c in cells)
    fam_cells = [c for c in cells if len(set(F.FAMILY[labels_np[c]])) <= 2]
    assert len(fam_cells) > 20, 'most replay cells must be confusion-group cells'


def fixture_v4(root):
    frame = pd.read_csv(root / 'data/train.csv', dtype=str)
    classes = sorted(frame.model.unique())
    stream = pd.DataFrame([dict(text=f'external fixture {c} {j}', model=c, prompt=f'pc:{j}') for c in classes for j in range(6)])
    stream.to_parquet(root / 'stream.parquet', index=False)
    ext = stream.copy(); ext['stream_row'] = np.arange(len(ext)); ext['label'] = np.searchsorted(classes, ext.model)
    ext['split'] = 'train_pool'
    ext.to_parquet(root / 'external.parquet', index=False)
    (root / 'external.json').write_text(json.dumps(dict(payload_sha256=sha(root / 'external.parquet'), stream_sha256=sha(root / 'stream.parquet'))))
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
            '--native-dir', str(root / 'native'), '--native-meta', str(root / 'meta'), '--token-cache', str(root / 'tokens'),
            '--targets', str(root / 'targets.npz'), '--stream', str(root / 'stream.parquet'), '--external', str(root / 'external.parquet'),
            '--fold', '0', '--deadline', future, '--device', 'cpu', '--max-batches', '4', '--micro-batch', '1',
            '--lr-lora', '1e-3', '--lr-head', '1e-2']


def worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    root = Path(directory); G.V2T.BASE = root
    common = arguments(root)
    original_step = torch.optim.AdamW.step
    calls = dict(n=0)
    def pause_after_two(opt, *a, **kw):
        value = original_step(opt, *a, **kw); calls['n'] += 1
        if calls['n'] == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return value
    torch.optim.AdamW.step = pause_after_two
    paused = common + ['--out', str(root / 'resumed')]
    try:
        G.main(paused, model_factory=model_factory)
    except SystemExit as error:
        assert error.code == 75
    else:
        raise AssertionError('Interrupted A1 training must pause')
    finally:
        torch.optim.AdamW.step = original_step
    state = torch.load(root / 'resumed/fold0/checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['state']['batch'] == 2
    G.main(paused, model_factory=model_factory)
    done = json.loads((root / 'resumed/fold0/done.json').read_text())
    assert done['method'] == 'v4_a1_hsc_native' and done['optimizer_steps'] == 4 and done['resumes'] == 1
    assert done['pool_cells'] >= 1 and done['protocol']['anchor_share'] == .75
    G.main(common + ['--out', str(root / 'straight')], model_factory=model_factory)
    a = load_file(str(root / 'resumed/fold0/trainables.safetensors')); b = load_file(str(root / 'straight/fold0/trainables.safetensors'))
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    pub = load_file(str(root / 'native/fold0/trainables.safetensors'))
    assert any(not torch.equal(v, pub[k]) for k, v in a.items())
    G.main(paused, model_factory=lambda *x: (_ for _ in ()).throw(AssertionError('retrained')))
    try:
        G.main(paused + ['--lambda-sib', '.3'], model_factory=model_factory)
    except ValueError as error:
        assert 'Completed A1' in str(error)
    else:
        raise AssertionError('Changed A1 protocol reused')


if __name__ == '__main__':
    unit()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v4_a1_') as directory:
        fixture(directory); fixture_v4(Path(directory))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, directory), nprocs=4, join=True)
    print(json.dumps(dict(a1_units=True, world_size=4, official_and_replay_cells=True, pause_resume_exact=True,
                          wise_export=True, completed_reused=True, protocol_guard=True)))
