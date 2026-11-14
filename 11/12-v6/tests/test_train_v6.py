"""v6 trainer on four Gloo ranks (CPU, tiny random 36-layer model, synthetic data): pretrain pause/resume equals an
uninterrupted run; finetune for two seeds; soup = exact mean of the seed states, public export layout and an
11/12-evaluate-compatible done.json; completed reuse and protocol guard. Code test only, not an 8B run."""
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
import train_v6 as G
sys.path.insert(1, str(HERE.parents[2] / '12/tests'))
from test_pipeline import model_factory, Tokenizer
from windows import write_cache, identity, sha

CLASSES = [f'class{c:02}' for c in range(18)]


def fixture(root):
    (root / 'data').mkdir(); (root / 'model').mkdir()
    rows = [dict(text=f'official {p} {c} text', prompt_id=f'p{p}', model=CLASSES[c]) for p in range(10) for c in range(18)]
    pd.DataFrame(rows).to_csv(root / 'data/train.csv', index=False)
    val = [i for i, r in enumerate(rows) if r['prompt_id'] in ('p0', 'p1')]
    (root / 'folds.json').write_text(json.dumps({'0': val}))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json', 'merges.txt', 'vocab.json'):
        (root / 'model' / name).write_text('{}')
    rng = np.random.default_rng(0)
    pool = pd.DataFrame([dict(text=f'pool {j} {c} words', model=CLASSES[c], prompt=f'pc:{j}', w=float(rng.uniform(.1, 2)))
                         for c in range(18) for j in range(6)])
    pool.to_parquet(root / 'stream.parquet', index=False)
    (root / 'stream.json').write_text(json.dumps(dict(payload_sha256=sha(root / 'stream.parquet'))))
    texts = np.concatenate([pd.read_csv(root / 'data/train.csv').text.values, pool.text.values])
    write_cache(texts, Tokenizer(), root / 'tokens', identity(root / 'data', root / 'stream.parquet', root / 'model'))


def args(root, stage, **kw):
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    out = ['--stage', stage, '--data', str(root / 'data'), '--model', str(root / 'model'), '--folds', str(root / 'folds.json'),
           '--stream', str(root / 'stream.parquet'), '--token-cache', str(root / 'tokens'), '--out', str(root / kw.pop('out', 'runs')),
           '--deadline', future, '--device', 'cpu', '--micro-batch', '1', '--batch', '16', '--presentations', '48',
           '--lr-adapter', '1e-3', '--lr-head', '1e-2']
    for k, v in kw.items():
        out += [f'--{k.replace("_", "-")}', str(v)]
    return out


def worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    root = Path(directory)
    step = torch.optim.AdamW.step
    def pause(opt, *a, **kw):
        v = step(opt, *a, **kw); os.kill(os.getpid(), signal.SIGTERM); return v
    torch.optim.AdamW.step = pause
    try:
        G.main(args(root, 'pretrain'), model_factory=model_factory)
    except SystemExit as e:
        assert e.code == 75
    else:
        raise AssertionError('pretrain must pause')
    finally:
        torch.optim.AdamW.step = step
    G.main(args(root, 'pretrain'), model_factory=model_factory)
    G.main(args(root, 'pretrain', out='straight'), model_factory=model_factory)
    a = torch.load(root / 'runs/pretrain/state.pt', weights_only=False); b = torch.load(root / 'straight/pretrain/state.pt', weights_only=False)
    assert set(a) == set(b) and all(torch.equal(v, b[k]) for k, v in a.items())
    assert json.loads((root / 'runs/pretrain/done.json').read_text())['resumes'] == 1
    for s in (42, 43):
        G.main(args(root, 'finetune', fold=0, seed=s), model_factory=model_factory)
    G.main(args(root, 'soup', fold=0, seeds='42,43'), model_factory=model_factory)
    G.main(args(root, 'pretrain'), model_factory=lambda *x: (_ for _ in ()).throw(AssertionError('retrained')))
    try:
        G.main(args(root, 'pretrain', smoothing=.1), model_factory=model_factory)
    except ValueError as e:
        assert 'Completed v6' in str(e)
    else:
        raise AssertionError('changed protocol reused')


if __name__ == '__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v6_train_') as d:
        root = Path(d); fixture(root)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, d), nprocs=4, join=True)
        s42 = torch.load(root / 'runs/finetune/fold0/seed42/state.pt', weights_only=False)
        s43 = torch.load(root / 'runs/finetune/fold0/seed43/state.pt', weights_only=False)
        pre = torch.load(root / 'runs/pretrain/state.pt', weights_only=False)
        assert any(not torch.equal(s42[k], s43[k]) for k in s42), 'seeds must differ'
        assert any(not torch.equal(s42[k], pre[k]) for k in s42), 'finetune must move weights'
        export = load_file(str(root / 'runs/fold0/trainables.safetensors'))
        assert len(export) == 524
        lora = [k for k in s42 if k.endswith('.a.weight')][0]
        name = lora.replace('backbone.', 'encoder.backbone.base_model.model.').replace('.a.weight', '.lora_A.default.weight')
        assert torch.allclose(export[name], ((s42[lora].double() + s43[lora].double()) / 2).float())
        assert torch.allclose(export['plain.weight'], ((s42['plain.weight'].double() + s43['plain.weight'].double()) / 2).float())
        done = json.loads((root / 'runs/fold0/done.json').read_text())
        assert done['args']['fold'] == 0 and done['export_sha256'] == sha(root / 'runs/fold0/trainables.safetensors')
        val, inner = np.load(root / 'runs/fold0/val_rows.npy'), np.load(root / 'runs/fold0/inner_rows.npy')
        assert len(val) == 36 and len(inner) == 36 and not set(val) & set(inner)
        assert np.load(root / 'runs/fold0/val_probs_final.npy').shape == (36, 18)
    print(json.dumps(dict(world_size=4, pretrain_pause_resume_exact=True, two_seed_finetune=True, soup_is_mean=True,
                          export_tensors=524, evaluate_compatible_done=True, completed_reused=True, protocol_guard=True)))
