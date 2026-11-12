"""Four CPU ranks verify the global objective and empty-rank handling; no pretrained model."""
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train as S
from distributed import Runtime, train_step


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    def pad(self, rows, **kwargs):
        x = torch.tensor(rows['input_ids'])
        return Batch(input_ids=x, attention_mask=torch.ones_like(x))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 6)
        self.plain = nn.Linear(6, 3)
        self.centered = nn.Linear(6, 3)
        self.proj = nn.Sequential(nn.Linear(6, 4))

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids).mean(1)
        return self.plain(x), self.centered(x), F.normalize(self.proj(x), dim=-1)


def vector(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in model.parameters()])


def worker(rank, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    rt = Runtime('cpu')
    try:
        args = SimpleNamespace(hard_k=2, lambda_hard=.5, lambda_con=.2,
                               temperature=.1, lambda_cent=.5)
        for n, micro, force_cache in [(n, m, force) for n in (16, 17, 2, 1)
                                      for m, force in [(8, False), (2, True)]]:
            args.force_cache = force_cache
            torch.manual_seed(123)
            model = Tiny()
            ids = {i: [i % 40 + 1, (i + 3) % 40 + 1] for i in range(n)}
            y = np.arange(n) % 3
            weights = np.linspace(.5, 1.5, n, dtype=np.float32)
            groups = np.arange(n) % 2
            batch = Tokenizer().pad({'input_ids': list(ids.values())})
            out = model(batch['input_ids'], batch['attention_mask'])
            loss, *_ = S.cell_loss(*out, torch.tensor(y), torch.tensor(weights),
                                   torch.tensor(groups), model.centered.bias, torch.zeros(3), args)
            loss.backward(); reference = vector(model).clone()
            model.zero_grad(set_to_none=True)
            train_step(model, Tokenizer(), ids, y, weights, groups, list(range(n)),
                       torch.device('cpu'), args, micro, torch.zeros(3), S.cell_loss, rt)
            actual = vector(model)
            assert torch.allclose(actual, reference, atol=2e-6, rtol=2e-5), (n, (actual-reference).abs().max())
            if rank == 0:
                print(json.dumps(dict(rows=n,world_size=4,path='cache' if force_cache else 'direct',global_gradient_max_error=float((actual-reference).abs().max()),passed=True)),flush=True)
    finally:
        rt.close()


if __name__ == '__main__':
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=4, join=True)
