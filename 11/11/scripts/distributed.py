"""One model across torchrun ranks, preserving the global cell loss.

Gradient-cache outputs are gathered in original row order. Every rank computes
the same full-cell loss (including cross-rank SupCon and prompt-group centering),
then replays only its rows. Trainable gradients are SUM-reduced in buckets;
the loss already contains global means, so averaging gradients would be wrong.
"""
from __future__ import annotations

import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, device='cuda'):
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.primary = self.rank == 0
        self.device = torch.device('cpu' if device == 'cpu' else f'cuda:{self.local_rank}')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group('nccl' if self.device.type == 'cuda' else 'gloo',
                                    timeout=timedelta(minutes=10))

    def scalar(self, value):
        t = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.broadcast(t, 0)
        return t.item()

    def any(self, value):
        t = torch.tensor(int(value), dtype=torch.int32, device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def gather_objects(self, value):
        if self.world == 1:
            return [value]
        out = [None] * self.world
        dist.all_gather_object(out, value)
        return out

    def checkpoint_rng(self):
        value = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                     python=random.getstate(), cuda=None)
        if self.device.type == 'cuda':
            value['cuda'] = torch.cuda.get_rng_state(self.device)
        return self.gather_objects(value)

    def restore_rng(self, states):
        if len(states) != self.world:
            raise ValueError('Resume requires the same number of ranks as the checkpoint')
        s = states[self.rank]
        torch.set_rng_state(s['torch'])
        np.random.set_state(s['numpy'])
        random.setstate(s['python'])
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(s['cuda'], self.device)


def _gather_rows(local, n, rt):
    """Gather strided row partitions, allowing unequal and empty ranks."""
    if rt.world == 1:
        return local
    width = (n + rt.world - 1) // rt.world
    padded = local.new_zeros((width, local.shape[-1]))
    padded[:len(local)] = local
    pieces = [torch.empty_like(padded) for _ in range(rt.world)]
    dist.all_gather(pieces, padded)
    out = local.new_empty((n, local.shape[-1]))
    for rank, piece in enumerate(pieces):
        rows = torch.arange(n, device=local.device)[rank::rt.world]
        out[rows] = piece[:len(rows)]
    return out


def sum_gradients(parameters, rt, bucket_bytes=16 << 20):
    """SUM of per-row gradients, with bounded communication buffers."""
    if rt.world == 1:
        return
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in bucket:
            count = p.numel()
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(flat[offset:offset + count].view_as(p))
            offset += count

    for p in parameters:
        if not p.requires_grad:
            continue
        count = p.numel() * p.element_size()
        if bucket and (size + count > bucket_bytes or p.dtype != bucket[0].dtype):
            flush(); bucket, size = [], 0
        bucket.append(p); size += count
    flush()


def _check_failure(error, rt):
    code = 0 if error is None else (1 if isinstance(error, torch.cuda.OutOfMemoryError) else 2)
    t = torch.tensor(code, dtype=torch.int32, device=rt.device)
    if rt.world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if t.item() == 1:
        raise torch.cuda.OutOfMemoryError('A rank exhausted memory; retry the global cell with a smaller micro batch')
    if t.item() == 2:
        raise RuntimeError(f'A rank failed during the distributed cell: {error!r}') from error


def train_step(model, tok, ids, y, weights, groups, cell, device, args, micro,
               log_prior, loss_fn, rt):
    """Exact global-cell gradient caching; no rank-local contrastive objective."""
    local_rows = list(cell[rt.rank::rt.world])
    chunks = [local_rows[i:i + micro] for i in range(0, len(local_rows), micro)]
    states, outputs = [], [[], [], []]
    before_cpu = torch.get_rng_state()
    before_cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None

    def encode(rows):
        return tok.pad({'input_ids': [ids[int(i)] for i in rows]}, padding=True,
                       return_tensors='pt').to(device)

    error = None
    try:
        with torch.no_grad():
            for rows in chunks:
                states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device)
                               if device.type == 'cuda' else None))
                e = encode(rows)
                result = model(e['input_ids'], e['attention_mask'])
                for target, value in zip(outputs, result):
                    target.append(value.float())
    except Exception as exc:
        error = exc
    try:
        _check_failure(error, rt)
        widths = [model.plain.out_features, model.centered.out_features, model.proj[-1].out_features]
        local = [torch.cat(values) if values else torch.empty((0, width), device=device)
                 for values, width in zip(outputs, widths)]
        global_out = [_gather_rows(value, len(cell), rt).detach().requires_grad_() for value in local]
        bias = model.centered.bias.detach().clone().requires_grad_()
        lab = torch.as_tensor(y[cell], device=device)
        w = torch.as_tensor(weights[cell], device=device, dtype=torch.float32)
        gid = torch.as_tensor(groups[cell], device=device)
        loss, *parts = loss_fn(*global_out, lab, w, gid, bias, log_prior, args)
        loss.backward()
        replay_error = None
        try:
            for rows, (cpu_state, cuda_state), start in zip(chunks, states, range(0, len(local_rows), micro)):
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                e = encode(rows)
                result = model(e['input_ids'], e['attention_mask'])
                positions = torch.arange(rt.rank, len(cell), rt.world, device=device)[start:start + len(rows)]
                grads = [value.grad[positions] for value in global_out]
                torch.autograd.backward([value.float() for value in result], grads)
            # The centered loss also depends directly on the head bias, outside
            # cached outputs. Its global contribution must be added exactly once.
            if bias.grad is not None:
                if model.centered.bias.grad is None:
                    model.centered.bias.grad = torch.zeros_like(model.centered.bias)
                model.centered.bias.grad.add_(bias.grad / rt.world)
        except Exception as exc:
            replay_error = exc
        _check_failure(replay_error, rt)
        sum_gradients(model.parameters(), rt)
        return [float(value.detach()) for value in parts]
    except Exception:
        torch.set_rng_state(before_cpu)
        if before_cuda is not None:
            torch.cuda.set_rng_state(before_cuda, device)
        raise


def predict(model, tok, ids, device, serial_predict, rt):
    ix = np.arange(rt.rank, len(ids), rt.world)
    local = serial_predict(model, tok, [ids[i] for i in ix], device) if len(ix) else np.empty((0, model.plain.out_features), np.float32)
    pieces = rt.gather_objects((ix, local))
    out = np.empty((len(ids), model.plain.out_features), np.float32)
    for indices, values in pieces:
        out[indices] = values
    return out
