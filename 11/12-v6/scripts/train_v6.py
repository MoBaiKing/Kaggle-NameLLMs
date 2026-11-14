"""v6 native expert: the public native recipe (CHEN Han study.json, LB-proven in the 0.684 notebook) re-run on
better data, plus a seed soup. Architecture = public native layout (Qwen3-Embedding-8B, 8 depths, LoRA r16 /
alpha 32 on every projection, plain + centred heads), so the Kaggle runtime is unchanged.

  --stage pretrain   shared by all folds (pool rows only, so no official label reaches any fold):
                     fresh LoRA/heads, label-balanced sampler weighted by data_v6.w, global batch 64,
                     adapter lr 2e-4, head lr 5e-4, label smoothing 0.05, centred coefficient 0, warm-up 10 % +
                     cosine over a fixed presentation budget (default 400k ~ the public 392k).
  --stage finetune   per fold and seed, from the pretrain state: official rows of the fold's training split minus
                     the 18 inner calibration prompts, cells of 4 complete prompt groups, 2 epochs, adapter lr 5e-5,
                     head lr 5e-4, centred coefficient 0.5, label smoothing 0.05 (public official stage).
  --stage soup       uniform average of the fold's seed states (same pretrain init), first-window validation
                     probabilities for export parity, public-layout export and an 11/12-compatible done.json, so
                     11/12 evaluate.py fits the public calibration fields on the inner rows and checks parity.
Every stage checkpoints and exits 75 at --deadline; rerun the same command to resume.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from shared import BASE, ROOT, V1
import train as S
from distributed import Runtime, _check_failure, train_step, predict as parallel_predict
from helpers import atomic_save, official_cells
from windows import TokenStore, identity as token_identity, sha
import objective_v3 as O
import data_v6 as DV

METHOD = 'v6_native_data_soup'


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pretrain', 'finetune', 'soup'], required=True)
    p.add_argument('--data', type=Path, default=BASE / 'data/original')
    p.add_argument('--model', type=Path, default=BASE / 'base_staged')
    p.add_argument('--folds', type=Path, default=BASE / 'data/folds.json')
    p.add_argument('--stream', type=Path, default=DV.DEFAULT)
    p.add_argument('--token-cache', type=Path, default=ROOT / 'cache/tokens_v6')
    p.add_argument('--out', type=Path, default=ROOT / 'runs_v6')
    p.add_argument('--fold', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', default='42,43,44', help='soup: seeds to average')
    p.add_argument('--deadline', required=True)
    p.add_argument('--presentations', type=int, default=400000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--groups-per-cell', type=int, default=4)
    p.add_argument('--official-epochs', type=int, default=2)
    p.add_argument('--inner-prompts', type=int, default=18)
    p.add_argument('--lr-adapter', type=float, default=None, help='default 2e-4 pretrain / 5e-5 finetune')
    p.add_argument('--lr-head', type=float, default=5e-4)
    p.add_argument('--smoothing', type=float, default=.05)
    p.add_argument('--cent-coef', type=float, default=None, help='default 0 pretrain / 0.5 finetune')
    p.add_argument('--lora-drop', type=float, default=.05)
    p.add_argument('--feat-drop', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=float, default=.1)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=0, help='tests/smoke only')
    p.add_argument('--ckpt-min', type=float, default=20.)
    p.add_argument('--mem-cap-gb', type=float, default=28.5)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.lr_adapter is None:
        a.lr_adapter = 2e-4 if a.stage == 'pretrain' else 5e-5
    if a.cent_coef is None:
        a.cent_coef = 0. if a.stage == 'pretrain' else .5
    if a.stage != 'pretrain' and a.fold not in range(5):
        p.error('--fold 0..4 is required for finetune/soup')
    if datetime.fromisoformat(a.deadline).tzinfo is None:
        p.error('--deadline must contain a timezone')
    return a


def loss_v6(plain, cent, z, lab, w, gid, bias, prior, args):
    lp = plain.float().log_softmax(-1)
    rows = (1 - args.smoothing) * -lp.gather(1, lab[:, None])[:, 0] - args.smoothing * lp.mean(-1)
    ce = (rows * w).sum() / w.sum().clamp_min(1e-6)
    cl = S.centred_loss(cent.float(), bias, lab, w, gid, prior) if args.cent_coef > 0 else cent.sum() * 0
    return ce + args.cent_coef * cl + z.sum() * 0, ce.detach(), cl.detach()


def balanced_cells(labels, weights, rows, n_cells, size, rng):
    """Label-balanced cells: every slot draws a label uniformly, then a row of that label proportional to w."""
    by = {}
    for c in np.unique(labels[rows]):
        r = rows[labels[rows] == c]
        by[int(c)] = (r, weights[r] / weights[r].sum())
    classes = np.array(sorted(by))
    out = []
    for _ in range(n_cells):
        picks = rng.choice(classes, size=size)
        out.append([int(rng.choice(by[int(c)][0], p=by[int(c)][1])) for c in picks])
    return out


def _protocol(args, extra):
    code = {p.name: sha(p) for p in [Path(__file__), Path(DV.__file__), V1 / 'scripts/train.py', V1 / 'scripts/distributed.py',
                                    V1 / 'scripts/windows.py', BASE / 'scripts/helpers.py']}
    keep = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            if k not in {'data', 'model', 'folds', 'stream', 'token_cache', 'out', 'deadline', 'device', 'ckpt_min',
                         'mem_cap_gb', 'micro_batch'}}
    protocol = dict(method=METHOD, args=keep, code=code, **extra)
    return protocol, hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def _run(args, rt, model_factory):
    log = (lambda v: print(json.dumps(v), flush=True)) if rt.primary else (lambda v: None)
    stop = dict(flag=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    if rt.device.type == 'cuda':
        total = torch.cuda.mem_get_info(rt.device)[1] / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1., args.mem_cap_gb / total), rt.device)
    tr = pd.read_csv(args.data / 'train.csv', dtype=str)
    classes = sorted(tr.model.unique()); n_off = len(tr)
    ex = pd.read_parquet(args.stream, columns=['text', 'model', 'prompt', 'w'])
    meta = json.loads(args.stream.with_suffix('.json').read_text())
    if sha(args.stream) != meta['payload_sha256']:
        raise ValueError('v6 stream changed after it was built')
    y = np.concatenate([np.searchsorted(classes, tr.model.values), np.searchsorted(classes, ex.model.values)])
    prompts = np.concatenate([tr.prompt_id.values.astype(str), ex.prompt.values.astype(str)])
    gid = pd.factorize(prompts)[0].astype(np.int64)
    weights = np.concatenate([np.ones(n_off), ex.w.values.astype(np.float64)])
    store = TokenStore(args.token_cache)
    if store.manifest['identity'] != token_identity(args.data, args.stream, args.model) or len(store) != len(y):
        raise ValueError('Token cache differs from the current data/stream/tokenizer')
    folds = {int(k): np.array(v) for k, v in json.loads(args.folds.read_text()).items()}
    base_extra = dict(inputs=store.manifest['identity'], stream=meta['payload_sha256'], folds=sha(args.folds))
    if args.stage == 'soup':
        return _soup(args, rt, model_factory, classes, y, prompts, store, folds, base_extra, log)
    if args.stage == 'pretrain':
        run = args.out / 'pretrain'
        pool_rows = np.arange(n_off, len(y))
        n_steps = args.presentations // args.batch
        cells = balanced_cells(y, weights, pool_rows, n_steps, args.batch, np.random.default_rng([args.seed, 600]))
        init = None
        extra = dict(base_extra)
    else:
        run = args.out / f'finetune/fold{args.fold}/seed{args.seed}'
        pre = args.out / 'pretrain/done.json'
        if not pre.exists():
            raise SystemExit('Pretrain stage has not finished')
        pre_done = json.loads(pre.read_text())
        if sha(args.out / 'pretrain/state.pt') != pre_done['state_sha256']:
            raise ValueError('Pretrain state changed')
        val = folds[args.fold]
        rest = np.setdiff1d(np.arange(n_off), val)
        inner_p = S.inner_split(prompts[rest], args.inner_prompts, 42)
        fit = np.array([r for r in rest if prompts[r] not in inner_p], int)
        cells = [c for e in range(args.official_epochs) for c in
                 official_cells(prompts, y, fit, args.groups_per_cell, 18, args.groups_per_cell * 18,
                                np.random.default_rng([args.seed, args.fold, 700 + e]))]
        n_steps = len(cells)
        init = args.out / 'pretrain/state.pt'
        extra = dict(base_extra, pretrain=pre_done['protocol_sha256'], fold=args.fold)
    if args.max_steps:
        cells = cells[:args.max_steps]; n_steps = len(cells)
    protocol, digest = _protocol(args, extra)
    if rt.primary:
        run.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    done_path, ck = run / 'done.json', run / 'checkpoint.pt'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done['protocol_sha256'] != digest or sha(run / 'state.pt') != done['state_sha256']:
            raise ValueError('Completed v6 stage inputs or state changed')
        log(dict(reused=True, stage=args.stage, run=str(run))); return
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    torch.manual_seed(args.seed)
    arch = SimpleNamespace(model=str(args.model), lora_drop=args.lora_drop, feat_drop=args.feat_drop, proj_dim=256)
    tok, model = model_factory(arch, len(classes), rt.device)
    model.proj.requires_grad_(False)
    if init is not None:
        missing, unexpected = model.load_state_dict(torch.load(init, map_location='cpu', weights_only=False), strict=False)
        if unexpected:
            raise ValueError(f'Unexpected tensors in pretrain state: {unexpected[:3]}')
    elif rt.world > 1:
        for p_ in model.parameters():
            if p_.requires_grad:
                torch.distributed.broadcast(p_.data, 0)
    lora = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and n.startswith('backbone.')]
    heads = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and not n.startswith('backbone.')]
    params = lora + heads
    opt = torch.optim.AdamW([dict(params=lora, lr=args.lr_adapter, base=args.lr_adapter),
                             dict(params=heads, lr=args.lr_head, base=args.lr_head)], weight_decay=args.weight_decay)
    st = dict(step=0, resumes=0, micro=args.micro_batch, rows=0)
    if ck.exists():
        saved = torch.load(ck, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest or saved['world_size'] != rt.world:
            raise ValueError('Resume needs the same v6 protocol and rank count')
        model.load_state_dict(saved['model'], strict=False); opt.load_state_dict(saved['opt'])
        st = saved['st']; st['resumes'] += 1; rt.restore_rng(saved['rng_states'])
    else:
        torch.manual_seed(args.seed + 1000 * rt.rank)

    def save():
        rng = rt.checkpoint_rng()
        if rt.primary:
            atomic_save(dict(model=S.trainable_state(model), opt=opt.state_dict(), st=st, rng_states=rng,
                             world_size=rt.world, protocol_sha256=digest), ck)
        rt.barrier()

    stage_id = 1 if args.stage == 'pretrain' else 2 + args.seed
    zeros = torch.zeros(len(classes), device=rt.device)
    w_train = weights if args.stage == 'finetune' else np.ones_like(weights)
    last = time.time()
    model.train()
    while st['step'] < n_steps:
        if rt.any(stop['flag'] or time.time() >= deadline):
            save(); raise SystemExit(75)
        cell = cells[st['step']]
        ids = {int(r): store.select(int(r), seed=args.seed, step=st['step'], stage=stage_id) for r in cell}
        f = O.lr_factor(st['step'], n_steps, args.warmup)
        for g in opt.param_groups:
            g['lr'] = g['base'] * f
        try:
            parts = train_step(model, tok, ids, y, w_train.astype(np.float32), gid, cell, rt.device, args,
                               st['micro'], zeros, loss_v6, rt)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            if st['micro'] == 1:
                raise
            st['micro'] = max(1, st['micro'] // 2)
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if rt.any(not torch.isfinite(norm).item()):
            raise FloatingPointError('Non-finite v6 gradient')
        opt.step(); opt.zero_grad(set_to_none=True)
        st['step'] += 1; st['rows'] += len(cell)
        if st['step'] % 25 == 0 or st['step'] == n_steps:
            log(dict(stage=args.stage, fold=args.fold, seed=args.seed, step=st['step'], of=n_steps, lr_factor=round(f, 4),
                     ce=round(parts[0], 4), cent=round(parts[1], 4), grad_norm=round(float(norm), 3)))
        if time.time() - last > 60 * args.ckpt_min:
            save(); last = time.time()
    save()
    if rt.primary:
        state = {k: v for k, v in S.trainable_state(model).items() if not k.startswith('proj.')}
        atomic_save(state, run / 'state.pt')
        done = dict(status='TRAINED', method=METHOD, stage=args.stage, fold=args.fold, seed=args.seed, steps=st['step'],
                    rows=st['rows'], resumes=st['resumes'], protocol=protocol, protocol_sha256=digest,
                    state_sha256=sha(run / 'state.pt'))
        tmp = done_path.with_suffix('.tmp'); tmp.write_text(json.dumps(done, indent=1) + '\n'); os.replace(tmp, done_path)
        ck.unlink(missing_ok=True)
    rt.barrier()


def _soup(args, rt, model_factory, classes, y, prompts, store, folds, base_extra, log):
    run = args.out / f'fold{args.fold}'
    seeds = [int(s) for s in args.seeds.split(',')]
    states, sources = [], {}
    for s in seeds:
        d = args.out / f'finetune/fold{args.fold}/seed{s}'
        done = json.loads((d / 'done.json').read_text())
        if sha(d / 'state.pt') != done['state_sha256'] or done['fold'] != args.fold:
            raise ValueError(f'Seed {s} state missing or changed')
        sources[str(s)] = done['state_sha256']
        states.append(torch.load(d / 'state.pt', map_location='cpu', weights_only=False))
    if any(set(st) != set(states[0]) for st in states):
        raise ValueError('Seed states differ in layout')
    soup = {k: torch.stack([st[k].double() for st in states]).mean(0).float() for k in states[0]}
    if rt.primary:
        run.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    arch = SimpleNamespace(model=str(args.model), lora_drop=0., feat_drop=0., proj_dim=256)
    tok, model = model_factory(arch, len(classes), rt.device)
    model.load_state_dict(soup, strict=False); model.eval()
    val = folds[args.fold]
    rest = np.setdiff1d(np.arange(len(pd.read_csv(args.data / 'train.csv', dtype=str))), val)
    inner_p = S.inner_split(prompts[rest], args.inner_prompts, 42)
    inner = np.array([r for r in rest if prompts[r] in inner_p], int)
    ids = [store.select(int(r), first=True) for r in val]
    pv = parallel_predict(model, tok, ids, rt.device, S.predict, rt)
    if rt.primary:
        np.save(run / 'val_probs_final.npy', pv); np.save(run / 'val_rows.npy', val); np.save(run / 'inner_rows.npy', inner)
        S.export_public(model, run / 'trainables.safetensors')
        protocol, digest = _protocol(args, dict(base_extra, seeds=sources))
        done = dict(status='TRAINED', method=METHOD, name='11/12-v6 data + soup', args={'fold': args.fold, 'seeds': seeds},
                    seed_states=sources, protocol=protocol, protocol_sha256=digest,
                    export_sha256=sha(run / 'trainables.safetensors'),
                    val_ba_first_window=float(np.mean([(pv[y[val] == c].argmax(1) == c).mean() for c in np.unique(y[val])])))
        (run / 'done.json').write_text(json.dumps(done, indent=1) + '\n')
        log(dict(soup=True, fold=args.fold, seeds=seeds, val_ba_first_window=round(done['val_ba_first_window'], 4)))
    rt.barrier()


def main(argv=None, model_factory=S.build_model):
    args = parse(argv); rt = Runtime(args.device)
    lock = None
    try:
        if rt.device.type == 'cuda' and rt.world != 4:
            raise ValueError('v6 runs on exactly four cooperating torchrun ranks')
        error = None
        if rt.primary:
            try:
                args.out.mkdir(parents=True, exist_ok=True)
                lock = (args.out / f'.{args.stage}_{args.fold}_{args.seed}.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception as exc:
                error = exc
        _check_failure(error, rt)
        _run(args, rt, model_factory)
    finally:
        if lock is not None:
            lock.close()
        rt.close()


if __name__ == '__main__':
    main()
