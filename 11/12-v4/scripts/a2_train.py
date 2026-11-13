"""A2 (CCL-Gen) training on four DDP ranks, external prompt-clean data only (external_v4 a2_train rows), so all
2,699 official rows remain clean held-out data for validation and for fitting B.

Per text: candidate set = true class + up to 3 siblings / fixed confusables (random order) + random others to
K_TRAIN; loss = generative NLL (per-token mean, true class) + LAMBDA_DISC * CE over the K_TRAIN sequence scores.
LoRA lr 2e-4, prefix lr 1e-3, AdamW wd 0, warm-up 3% + cosine, global batch 32 texts, one epoch, bf16 base with
fp32 adapters, gradient checkpointing, checkpoint/resume (exit 75 at the deadline).
"""
import argparse
import hashlib
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from shared import ROOT
from oof_targets import sha
import ccl_gen as G
import external_v4 as E
import families as F

K_TRAIN = 4
METHOD = 'v4_a2_ccl_gen'


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True, help='Qwen3-1.7B-Base (HF revision pinned in experiment_plan.json)')
    p.add_argument('--external', type=Path, default=E.DEFAULT)
    p.add_argument('--out', type=Path, default=ROOT / 'runs_v4/a2')
    p.add_argument('--deadline', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--micro', type=int, default=4, help='Texts per rank per micro step')
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--lr-lora', type=float, default=2e-4)
    p.add_argument('--lr-prefix', type=float, default=1e-3)
    p.add_argument('--lambda-disc', type=float, default=1.)
    p.add_argument('--max-tokens', type=int, default=G.MAX_TOKENS)
    p.add_argument('--seed', type=int, default=42, choices=[42])
    p.add_argument('--max-steps', type=int, default=0, help='Smoke/tests only')
    p.add_argument('--ckpt-every', type=int, default=100)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--base-factory', default='', help='tests only')
    a = p.parse_args(argv)
    if datetime.fromisoformat(a.deadline).tzinfo is None:
        p.error('--deadline must contain a timezone')
    return a


def candidate_sets(labels, rng):
    out = np.zeros((len(labels), K_TRAIN), np.int64)
    n = len(F.CLASSES)
    for i, y in enumerate(labels):
        rivals = list(rng.permutation(np.flatnonzero(F.NEIGHBORS[y])))[:K_TRAIN - 1]
        others = [c for c in rng.permutation(n) if c != y and c not in rivals]
        out[i] = [y] + rivals + others[:K_TRAIN - 1 - len(rivals)]
    return out


def lr_factor(step, total, warm_frac=.03):
    warm = max(1, math.ceil(warm_frac * total))
    if step < warm:
        return (step + 1) / warm
    return .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main(argv=None, factory=None):
    a = parse(argv)
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if a.device == 'cuda' else 'cpu')
    if a.device == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if a.device == 'cuda' else 'gloo')
    primary = rank == 0
    log = (lambda v: print(json.dumps(v), flush=True)) if primary else (lambda v: None)
    stop = dict(flag=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    any_ = lambda v: (dist.all_reduce(t := torch.tensor(int(v), device=device), op=dist.ReduceOp.MAX) or bool(t.item())) if world > 1 else bool(v)
    try:
        external = pd.read_parquet(a.external)
        meta_ext = json.loads(a.external.with_suffix('.json').read_text())
        if sha(a.external) != meta_ext['payload_sha256']:
            raise ValueError('External v4 set changed')
        train = external[external.a2_train].reset_index(drop=True)
        protocol = dict(method=METHOD, external=meta_ext['payload_sha256'], world=world, k_train=K_TRAIN,
                        rank=G.RANK, alpha=G.ALPHA, prefix=G.PREFIX,
                        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
                              if k not in {'out', 'deadline', 'device', 'ckpt_every', 'base_factory', 'external'}},
                        code={n: sha(Path(__file__).with_name(n)) for n in ['a2_train.py', 'ccl_gen.py', 'families.py']})
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        a.out.mkdir(parents=True, exist_ok=True)
        done_path, ckpt = a.out / 'done.json', a.out / 'checkpoint.pt'
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done['protocol_sha256'] != digest or sha(a.out / 'adapters.safetensors') != done['adapters_sha256']:
                raise ValueError('Completed A2 run inputs or artifacts changed')
            log(dict(reused=True, status='TRAINED')); return
        deadline = datetime.fromisoformat(a.deadline).timestamp()
        torch.manual_seed(a.seed)
        dtype = torch.bfloat16 if a.device == 'cuda' else torch.float32
        tok, model = (factory or G.build)(str(a.base), len(F.CLASSES), device, dtype, F.CLASSES)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.lm.train()
        lora = [p for n, p in model.named_parameters() if p.requires_grad and n != 'prefix']
        opt = torch.optim.AdamW([dict(params=lora, lr=a.lr_lora, base=a.lr_lora),
                                 dict(params=[model.prefix], lr=a.lr_prefix, base=a.lr_prefix)], weight_decay=0.)
        rng = np.random.default_rng([a.seed, 777])
        order = np.concatenate([rng.permutation(len(train)) for _ in range(a.epochs)])
        cands = candidate_sets(train.label.values[order], np.random.default_rng([a.seed, 778]))
        global_batch = world * a.micro * a.accum
        total = len(order) // global_batch
        if a.max_steps:
            total = min(total, a.max_steps)
        state = dict(step=0, resumes=0)
        if ckpt.exists():
            saved = torch.load(ckpt, map_location='cpu', weights_only=False)
            if saved['protocol_sha256'] != digest or saved['world'] != world:
                raise ValueError('Resume needs the same A2 protocol and rank count')
            model.load_trainable(saved['model']); opt.load_state_dict(saved['opt'])
            state = saved['state']; state['resumes'] += 1
        encoded = {}
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def save():
            if primary:
                tmp = str(ckpt) + '.tmp'
                torch.save(dict(model=model.trainable_state(), opt=opt.state_dict(), state=state, world=world,
                                protocol_sha256=digest), tmp)
                os.replace(tmp, ckpt)
            if world > 1:
                dist.barrier()

        params = lora + [model.prefix]
        while state['step'] < total:
            if any_(stop['flag'] or time.time() >= deadline):
                save(); raise SystemExit(75)
            f = lr_factor(state['step'], total)
            for g in opt.param_groups:
                g['lr'] = g['base'] * f
            stats = []
            for micro in range(a.accum):
                start = state['step'] * global_batch + (micro * world + rank) * a.micro
                idx = order[start:start + a.micro]; cand = cands[start:start + a.micro]
                for i in idx:
                    if i not in encoded:
                        encoded[i] = G.encode(tok, [train.text.values[i]], a.max_tokens)[0]
                seqs = [encoded[i] for i in idx for _ in range(K_TRAIN)]
                ids, mask = G.pad(seqs, pad_id, device)
                classes = torch.tensor(cand.reshape(-1), device=device)
                sums, n = model.sequence_scores(ids, mask, classes)
                sums = sums.float().reshape(len(idx), K_TRAIN); n = n.float().reshape(len(idx), K_TRAIN)[:, 0]
                gen = (-sums[:, 0] / n).mean()
                disc = torch.nn.functional.cross_entropy(sums, torch.zeros(len(idx), dtype=torch.long, device=device))
                loss = (gen + a.lambda_disc * disc) / a.accum
                loss.backward()
                stats.append([float(gen), float(disc), float((sums.argmax(1) == 0).float().mean())])
            if world > 1:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad); p.grad /= world
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if any_(not torch.isfinite(norm).item()):
                raise FloatingPointError('Non-finite A2 gradient')
            opt.step(); opt.zero_grad(set_to_none=True)
            state['step'] += 1
            if state['step'] % a.ckpt_every == 0:
                save()
            if state['step'] % 10 == 0 or state['step'] == total:
                m = np.mean(stats, 0)
                log(dict(step=state['step'], total=total, lr_factor=round(f, 4), gen_nll=round(m[0], 4),
                         disc_ce=round(m[1], 4), cand_acc=round(m[2], 3), grad_norm=round(float(norm), 3)))
        save()
        if primary:
            from safetensors.torch import save_file
            save_file(model.trainable_state(), str(a.out / 'adapters.safetensors'))
            bundle = dict(method=METHOD, classes=F.CLASSES, rank=G.RANK, alpha=G.ALPHA, prefix=G.PREFIX,
                          max_tokens=a.max_tokens, probe_ids=G.encode(tok, [G.PROBE])[0],
                          base={n: sha(Path(a.base) / n) for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')
                                if (Path(a.base) / n).exists()},
                          adapters_sha256=sha(a.out / 'adapters.safetensors'))
            (a.out / 'a2_bundle.json').write_text(json.dumps(bundle, indent=1) + '\n')
            done = dict(status='TRAINED', method=METHOD, steps=state['step'], total=total, resumes=state['resumes'],
                        rows=int(len(train)), protocol=protocol, protocol_sha256=digest,
                        adapters_sha256=bundle['adapters_sha256'], bundle_sha256=sha(a.out / 'a2_bundle.json'))
            (a.out / 'done.json').write_text(json.dumps(done, indent=1) + '\n')
            ckpt.unlink(missing_ok=True)
        if world > 1:
            dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
