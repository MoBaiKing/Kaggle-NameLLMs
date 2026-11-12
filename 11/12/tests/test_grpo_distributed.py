"""Four CPU ranks compare full-window GRPO gradients to a dense single model."""
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed import Runtime, train_step
from grpo import aggregate_windows, native_log_probs, objective, rollout
from test_distributed import Tiny, Tokenizer, vector
from test_grpo import CAL


def worker(rank, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1); rt = Runtime('cpu')
    try:
        for owners_list in [[0], [0, 1], [0, 0, 1, 2, 2, 2, 3, 4, 4]]:
            n, texts = len(owners_list), max(owners_list) + 1
            owners = torch.tensor(owners_list); weights = torch.arange(1., n + 1)
            labels = torch.arange(texts) % 3; groups = torch.arange(texts) // 2
            balance = torch.linspace(.5, 1.5, texts)
            reduce = lambda x: aggregate_windows(x, owners, weights, texts)
            for force_cache in [False, True]:
                torch.manual_seed(12); model = Tiny().eval()
                ids = {i: [i + 1, i + 2] for i in range(n)}
                e = Tokenizer().pad({'input_ids': list(ids.values())})
                p, c, z = model(**e)
                lp = native_log_probs(reduce(p), reduce(c), model.centered.bias, groups, CAL)
                samples = rollout(lp, labels, 64, 19)
                ref = torch.randn_like(lp).log_softmax(-1).detach()
                # Reuse the same old rollout for a later policy iterate, so the
                # multi-update clipping branch is exercised as well.
                with torch.no_grad(): model.plain.weight.add_(.2)
                def loss_fn(p, c, z, lab, w, gid, bias, prior, args):
                    current = native_log_probs(reduce(p), reduce(c), bias, groups, CAL)
                    loss, parts = objective(current, ref, labels, balance, samples)
                    return loss + z.sum() * 0, *parts
                out = model(**e)
                loss_fn(*out, None, None, None, model.centered.bias, None, None)[0].backward()
                expected = vector(model).clone(); model.zero_grad(set_to_none=True)
                args = SimpleNamespace(force_cache=force_cache)
                train_step(model, Tokenizer(), ids, np.zeros(n, int), np.ones(n, np.float32),
                           np.zeros(n, int), list(range(n)), torch.device('cpu'), args,
                           1 if force_cache else 8, torch.zeros(3), loss_fn, rt)
                error = (vector(model) - expected).abs().max().item()
                assert torch.allclose(vector(model), expected, atol=2e-6, rtol=2e-5), error
                if rank == 0:
                    print(json.dumps(dict(world=4, windows=n, texts=texts, cache=force_cache,
                                          gradient_max_error=error, passed=True)), flush=True)
    finally:
        rt.close()


if __name__ == '__main__':
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=4, join=True)
