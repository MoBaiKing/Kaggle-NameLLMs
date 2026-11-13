"""A2 CCL-Gen on CPU with a tiny random Qwen3 causal LM: score = exact sum of next-token log-probs after the class
prefix; state round trip; candidate sets; four-rank DDP training with pause/resume == uninterrupted; OOF scoring
and bundle; Kaggle worker entry point. Code test, not a 1.7B run."""
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
import ccl_gen as G
import a2_train as A2
import a2_score as SC
import families as F
from oof_targets import sha


class Tok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [3 + (ord(ch) % 250) for ch in str(text)]
        return {'input_ids': ids[:max_length] if truncation and max_length else ids}


def factory(base, n_classes, device, dtype, names=None):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    cfg._attn_implementation = 'sdpa'
    lm = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    torch.manual_seed(42)
    return Tok(), G.CCLGen(lm, n_classes).to(device)


def unit():
    tok, model = factory(None, 18, torch.device('cpu'), torch.float32)
    for m in model.lm.modules():
        if isinstance(m, G.LoRALinear):
            torch.nn.init.normal_(m.b.weight, std=.02)
    text = G.encode(tok, ['hello sibling world'], 64)[0]
    ids, mask = G.pad([text, text[:5]], 0, torch.device('cpu'))
    s, n = model.sequence_scores(ids, mask, torch.tensor([3, 7]))
    pre = model.prefix[3][None]; emb = model.lm.get_input_embeddings()(ids[:1])
    logits = model.lm(inputs_embeds=torch.cat([pre, emb], 1)).logits[0].log_softmax(-1)
    P = pre.shape[1]
    manual = sum(float(logits[P - 1 + t, text[t]]) for t in range(len(text)))
    assert abs(float(s[0]) - manual) < 1e-4 and int(n[1]) == 5
    state = model.trainable_state(); _, other = factory(None, 18, torch.device('cpu'), torch.float32)
    other.load_trainable(state)
    assert all(torch.equal(v, other.trainable_state()[k]) for k, v in state.items())
    c = A2.candidate_sets(np.array([F.CLASSES.index('gemini-3.7-flash'), F.CLASSES.index('qwen3.8-max')]), np.random.default_rng(0))
    assert c[0, 0] == F.CLASSES.index('gemini-3.7-flash') and all(F.NEIGHBORS[c[0, 0], j] for j in c[0, 1:])
    assert len(set(c[1])) == A2.K_TRAIN and c[1, 0] == F.CLASSES.index('qwen3.8-max')


def fixture(root):
    rows = []
    for ci, c in enumerate(F.CLASSES):
        for j in range(8):
            rows.append(dict(stream_row=len(rows), text=f'{c} writes passage {j} ' * (1 + ci % 3), model=c, label=ci,
                             prompt=f'pc:{j}', split='train_pool' if j < 6 else 'dev_pool', a2_train=j < 6, a2_dev=j >= 6))
    ext = pd.DataFrame(rows); ext.to_parquet(root / 'external.parquet', index=False)
    (root / 'external.json').write_text(json.dumps(dict(payload_sha256=sha(root / 'external.parquet'))))


def args(root, out):
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    return ['--base', str(root / 'nobase'), '--external', str(root / 'external.parquet'), '--out', str(root / out),
            '--deadline', future, '--device', 'cpu', '--micro', '1', '--accum', '1', '--max-steps', '3', '--max-tokens', '48']


def worker(rank, port, directory):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    root = Path(directory)
    step = torch.optim.AdamW.step
    def pause(opt, *a, **kw):
        v = step(opt, *a, **kw); os.kill(os.getpid(), signal.SIGTERM); return v
    torch.optim.AdamW.step = pause
    try:
        A2.main(args(root, 'resumed'), factory=factory)
    except SystemExit as e:
        assert e.code == 75
    else:
        raise AssertionError('A2 must pause')
    finally:
        torch.optim.AdamW.step = step
    A2.main(args(root, 'resumed'), factory=factory)
    A2.main(args(root, 'straight'), factory=factory)
    a = load_file(str(root / 'resumed/adapters.safetensors')); b = load_file(str(root / 'straight/adapters.safetensors'))
    assert all(torch.allclose(v, b[k], atol=1e-6) for k, v in a.items())
    done = json.loads((root / 'resumed/done.json').read_text())
    assert done['steps'] == 3 and done['resumes'] == 1
    try:
        A2.main(args(root, 'resumed') + ['--lambda-disc', '.5'], factory=factory)
    except ValueError as e:
        assert 'Completed A2' in str(e)
    else:
        raise AssertionError('Changed A2 protocol reused')


if __name__ == '__main__':
    unit()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v4_a2_') as d:
        root = Path(d); fixture(root)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, d), nprocs=4, join=True)
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            os.environ.pop(name, None)
        SC.main(['--base', str(root / 'nobase'), '--run', str(root / 'resumed'), '--external', str(root / 'external.parquet'),
                 '--smoke', '6', '--device', 'cpu'], factory=factory)
        rec = json.loads((root / 'resumed/a2_scores_smoke.json').read_text())
        with np.load(root / 'resumed/a2_scores_smoke.npz') as z:
            assert z['official_sums'].shape == (6, 18) and z['official_done'].all() and np.isfinite(z['official_sums']).all()
        assert rec['adapters_sha256'] == json.loads((root / 'resumed/a2_bundle.json').read_text())['adapters_sha256']
    print(json.dumps(dict(a2_units=True, score_equals_manual_logprob=True, ddp_world_size=4, pause_resume_match=True,
                          protocol_guard=True, oof_scoring=True)))
