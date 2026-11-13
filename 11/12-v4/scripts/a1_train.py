"""A1 (HSC-Native): four-rank continuation of one public native fold state.

v3 engine and objective (OOF-weighted ensemble-aware CE + KL to the public native, frozen public calibration, WiSE
alpha 0.5 export), but against the LB-anchored proxy policy (proxy_v4: native share 0.75, OOF weights from
oof_targets_v4), plus:
  * sibling-structured replay cells from the prompt-clean external pool (external_v4), alternating 1:1 with the
    official cells; pool cells use the native's own label-smoothed CE (no OOF parent exists for them);
  * hierarchical supervised contrastive loss on the native projection (graded family positives);
  * sibling / confusable margin loss on the calibrated native log-probabilities.
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

from shared import BASE, ROOT, V1, V2, V3, load_module
import train as S
from distributed import Runtime, _check_failure, train_step
from grpo import aggregate_windows, native_log_probs
from helpers import atomic_save, official_cells
from windows import TokenStore, write_cache, sha, identity as stream_identity
import a1_objective as A
import external_v4 as E
import oof_targets_v4 as T4
import proxy_v4 as X4
import objective_v3 as O
import oof_targets as T
import wise as W

V2T = load_module('v2_grpo_train', V2 / 'scripts/grpo_train.py')
NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'
META = BASE / 'external/native_q8_meta/ntl-native-q8-multidepth-fivefold-0679'
METHOD = 'v4_a1_hsc_native'


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=BASE / 'data/original')
    p.add_argument('--model', type=Path, default=BASE / 'base_staged')
    p.add_argument('--folds', type=Path, default=BASE / 'data/folds.json')
    p.add_argument('--native-dir', type=Path, default=NATIVE)
    p.add_argument('--native-meta', type=Path, default=META)
    p.add_argument('--token-cache', type=Path, default=ROOT / 'cache/tokens_stream')
    p.add_argument('--stream', type=Path, default=BASE / 'data/stream_8b_cap9500.parquet')
    p.add_argument('--external', type=Path, default=E.DEFAULT)
    p.add_argument('--pool-ratio', type=int, default=1, help='Replay cells per official cell')
    p.add_argument('--lambda-sib', type=float, default=.2)
    p.add_argument('--lambda-con', type=float, default=.1)
    p.add_argument('--family-pos', type=float, default=.25)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--margin', type=float, default=1.)
    p.add_argument('--targets', type=Path, default=T4.DEFAULT)
    p.add_argument('--out', type=Path, default=ROOT / 'runs_v4/a1')
    p.add_argument('--fold', type=int, required=True, choices=range(5))
    p.add_argument('--deadline', required=True, help='ISO time; checkpoint and exit 75 at deadline')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=16, help='GLOBAL text batch, shared by all four ranks')
    p.add_argument('--micro-batch', type=int, default=2, help='Windows per rank, gradient cached')
    p.add_argument('--lr-lora', type=float, default=1e-5)
    p.add_argument('--lr-head', type=float, default=1e-4)
    p.add_argument('--warmup', type=float, default=.1)
    p.add_argument('--smoothing', type=float, default=.05)
    p.add_argument('--beta', type=float, default=.05)
    p.add_argument('--seed', type=int, default=42, choices=[42])
    p.add_argument('--max-batches', type=int, default=0, help='Smoke/tests only; use a separate output')
    p.add_argument('--ckpt-every', type=int, default=20)
    p.add_argument('--mem-cap-gb', type=float, default=28.5)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU only for synthetic tests')
    a = p.parse_args(argv)
    if min(a.lambda_sib, a.lambda_con, a.family_pos, a.margin) < 0 or a.temperature <= 0 or a.pool_ratio < 0:
        p.error('Invalid A1 auxiliary coefficients')
    if min(a.epochs, a.batch, a.micro_batch, a.ckpt_every) < 1 or a.max_batches < 0:
        p.error('Invalid batch/epoch count')
    if min(a.lr_lora, a.lr_head, a.mem_cap_gb) <= 0 or not 0 <= a.warmup < 1 or not 0 <= a.smoothing < 1 or a.beta < 0:
        p.error('Invalid learning rate, schedule, smoothing, KL weight or budget')
    if datetime.fromisoformat(a.deadline).tzinfo is None:
        p.error('--deadline must contain a timezone')
    return a


def _run(args, rt, model_factory, verifier):
    run = args.out / f'fold{args.fold}'
    if rt.primary:
        run.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    stop = dict(flag=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    log = lambda value: print(json.dumps(value), flush=True) if rt.primary else None
    if rt.device.type == 'cuda':
        total = torch.cuda.mem_get_info(rt.device)[1] / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1., args.mem_cap_gb / total), rt.device)
    frame = pd.read_csv(args.data / 'train.csv', dtype=str)
    classes = sorted(frame.model.unique())
    y = np.searchsorted(classes, frame.model.values)
    groups = pd.factorize(frame.prompt_id)[0]
    identity = V2T.token_identity(args)
    fit, val, calibration, source_hashes = verifier(args, frame, identity)
    stream_id = stream_identity(args.data, args.stream, args.model)
    external = pd.read_parquet(args.external)
    external_meta = json.loads(args.external.with_suffix('.json').read_text())
    if sha(args.external) != external_meta['payload_sha256'] or external_meta['stream_sha256'] != stream_id['stream']:
        raise ValueError('External v4 set does not match the pinned stream')
    pool_rows = external.index[external.split == 'train_pool'].to_numpy()
    pool_label = external.label.to_numpy(); pool_prompt = external.prompt.astype(str).to_numpy()
    pool_stream_row = external.stream_row.to_numpy()
    parent, oof_weight, targets_meta = T.load_verified(args.targets, args.data / 'train.csv', args.folds, len(frame))
    code_files = {n: Path(__file__).with_name(n) for n in ['a1_train.py', 'a1_objective.py', 'families.py', 'external_v4.py', 'shared.py',
                                                          'proxy_v4.py', 'oof_targets_v4.py']}
    code_files.update({f'v3/{n}': V3 / 'scripts' / n for n in ['objective_v3.py', 'oof_targets.py', 'proxy_v3.py', 'wise.py']})
    code_files['v2/grpo_train.py'] = V2 / 'scripts/grpo_train.py'
    code_files.update({f'v1/{n}': V1 / 'scripts' / n for n in ['grpo.py', 'train.py', 'distributed.py', 'windows.py']})
    code_files['helpers.py'] = Path(official_cells.__code__.co_filename)
    protocol = dict(method=METHOD, inputs=identity, stream=stream_id, external=external_meta['payload_sha256'],
                    folds=sha(args.folds), public=source_hashes,
                    calibration=calibration, anchor_share=X4.ANCHOR_SHARE, targets=dict(payload_sha256=targets_meta['payload_sha256'],
                                                          eps=targets_meta['eps'], gamma=targets_meta['gamma'],
                                                          parent_temperature=targets_meta['parent_temperature']),
                    execution=dict(torch=torch.__version__, world_size=rt.world, device=rt.device.type, attention='sdpa'),
                    arguments={k: v for k, v in vars(args).items() if k not in
                               {'data', 'model', 'folds', 'native_dir', 'native_meta', 'token_cache', 'targets', 'stream', 'external',
                                'out', 'deadline', 'device', 'ckpt_every', 'mem_cap_gb'}},
                    code={n: sha(path) for n, path in code_files.items()})
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    done_path, checkpoint = run / 'done.json', run / 'checkpoint.pt'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done['protocol_sha256'] != digest or any(sha(run / n) != h for n, h in done['artifact_sha256'].items()):
            raise ValueError('Completed A1 run inputs or artifacts changed')
        log(dict(reused=True, status='TRAINED', fold=args.fold))
        return
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if rt.scalar(time.time()) >= deadline:
        raise SystemExit(75)
    torch.manual_seed(args.seed)
    architecture = SimpleNamespace(model=str(args.model), lora_drop=0., feat_drop=0., proj_dim=256)
    tok, model = model_factory(architecture, len(classes), rt.device)
    public_state = args.native_dir / f'fold{args.fold}/trainables.safetensors'
    V2T.load_public(model, public_state)
    cache_error = None
    if rt.primary:
        try:
            stream_text = pd.read_parquet(args.stream, columns=['text']).text.values
            write_cache(np.concatenate([frame.text.values, stream_text]), tok, args.token_cache, stream_id)
        except Exception as exc:
            cache_error = exc
    _check_failure(cache_error, rt)
    store = TokenStore(args.token_cache)
    if store.manifest['identity'] != stream_id or len(store) <= len(frame):
        raise ValueError('Token cache inputs differ')
    sequences, starts, lengths, window_weights = [], {}, {}, []
    for row in fit:
        seqs, weights = store.all_windows(int(row))
        starts[int(row)], lengths[int(row)] = len(sequences), len(seqs)
        sequences.extend(seqs); window_weights.extend(weights)
    official_cells_ = [cell for epoch in range(args.epochs) for cell in
                       official_cells(frame.prompt_id.values, y, fit, 4, 4, args.batch,
                                      np.random.default_rng([args.seed, epoch, 330]))]
    replay = A.pool_cells(pool_label, pool_prompt, pool_rows, args.pool_ratio * len(official_cells_), args.batch,
                          np.random.default_rng([args.seed, 331]))
    pool_start = {}
    for cell in replay:
        for r in cell:
            if r not in pool_start:
                pool_start[r] = len(sequences)
                sequences.append(store.select(len(frame) + int(pool_stream_row[r]), first=True)); window_weights.append(1.)
    window_weights = np.asarray(window_weights, np.float32)
    model.proj.requires_grad_(True)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    if not lora or not heads or not any(n.startswith('proj.') for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError('Unexpected trainable parameter set')
    params = lora + heads
    opt = torch.optim.AdamW([dict(params=lora, lr=args.lr_lora, base_lr=args.lr_lora),
                             dict(params=heads, lr=args.lr_head, base_lr=args.lr_head)], weight_decay=0.)
    batches = []
    for i, cell in enumerate(official_cells_):
        batches.append(('official', cell))
        batches.extend(('pool', c) for c in replay[i * args.pool_ratio:(i + 1) * args.pool_ratio])
    if args.max_batches:
        batches = batches[:args.max_batches]
    state = dict(batch=0, optimizer_steps=0, resumes=0, micro=args.micro_batch)
    saved = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest or saved['world_size'] != rt.world:
            raise ValueError('Resume needs the same v3 protocol and rank count')
        state = saved['state']; state['resumes'] += 1
    # Frozen public native outputs on every training window (KL anchor), computed once per run directory.
    ref_path, ref_record = run / 'reference.npz', run / 'reference.json'
    if ref_path.exists():
        record = json.loads(ref_record.read_text())
        if record['protocol_sha256'] != digest or record['sha256'] != sha(ref_path):
            raise ValueError('Frozen public reference cache changed')
        with np.load(ref_path, allow_pickle=False) as z:
            reference = torch.tensor(z['logits'], device=rt.device)
    else:
        if saved is not None:
            raise ValueError('Resume is missing its frozen public reference')
        while True:
            try:
                reference = V2T.head_logits(model, tok, sequences, rt, state['micro'],
                                            lambda: stop['flag'] or time.time() >= deadline)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if state['micro'] == 1:
                    raise
                state['micro'] = max(1, state['micro'] // 2)
        if rt.primary:
            with ref_path.with_suffix('.tmp').open('wb') as f:
                np.savez(f, logits=reference.cpu().numpy(), protocol_sha256=digest)
            os.replace(ref_path.with_suffix('.tmp'), ref_path)
            temp = ref_record.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(protocol_sha256=digest, sha256=sha(ref_path))) + '\n')
            os.replace(temp, ref_record)
        rt.barrier()
    if reference.shape != (len(sequences), 2 * len(classes)) or not torch.isfinite(reference).all():
        raise ValueError('Invalid frozen reference logits')
    reference_sha = sha(ref_path)
    ref_bias = model.centered.bias.detach().clone()
    if saved is not None:
        model.load_state_dict(saved['model'], strict=False)
        opt.load_state_dict(saved['opt']); rt.restore_rng(saved['rng_states'])
    counts = np.bincount(y[fit], minlength=len(classes))
    if (counts == 0).any():
        raise ValueError('Public training split is missing a class')
    class_weights = len(fit) / (len(classes) * counts)
    row_weight = class_weights[y] * oof_weight
    weight_scale = float(row_weight[fit].mean())

    def save():
        rng = rt.checkpoint_rng()
        if rt.primary:
            atomic_save(dict(model=S.trainable_state(model), opt=opt.state_dict(), state=state,
                             rng_states=rng, world_size=rt.world, protocol_sha256=digest,
                             reference_sha256=reference_sha), checkpoint)
        rt.barrier()

    save()
    win_y = np.zeros(len(sequences), np.int64)
    win_w = np.ones(len(sequences), np.float32)
    ids = {i: seq for i, seq in enumerate(sequences)}
    while state['batch'] < len(batches):
        if rt.any(stop['flag']) or rt.scalar(time.time()) >= deadline:
            save(); raise SystemExit(75)
        kind, cell = batches[state['batch']]
        indices, owners = [], []
        if kind == 'official':
            for owner, row in enumerate(cell):
                indices.extend(range(starts[row], starts[row] + lengths[row]))
                owners.extend([owner] * lengths[row])
            labels_np, groups_np = y[cell], groups[cell]
        else:
            indices = [pool_start[r] for r in cell]; owners = list(range(len(cell)))
            labels_np, groups_np = pool_label[cell], -1 - np.arange(len(cell))
        owners = torch.tensor(owners, device=rt.device)
        weights = torch.tensor(window_weights[indices], device=rt.device)
        labels = torch.tensor(labels_np, device=rt.device)
        group = torch.tensor(groups_np, device=rt.device)
        reduce = lambda logits, o=owners, w=weights, n=len(cell): aggregate_windows(logits, o, w, n)
        rp, rc = reference[indices].chunk(2, -1)
        ref_lp = native_log_probs(reduce(rp), reduce(rc), ref_bias, group, calibration).detach()
        if kind == 'official':
            cell_weight = torch.tensor(row_weight[cell], dtype=torch.float32, device=rt.device)
            parent_lp = torch.tensor(parent[cell], device=rt.device)

        def loss_fn(plain, centered, z, lab, w, gid, bias, prior, unused, kind=kind):
            lp = native_log_probs(reduce(plain), reduce(centered), bias, group, calibration)
            text_z = torch.nn.functional.normalize(reduce(z), dim=-1)
            if kind == 'official':
                main, parts = A.ensemble_objective(lp, ref_lp, parent_lp, labels, cell_weight, weight_scale,
                                                   X4.ANCHOR_SHARE, args.smoothing, args.beta)
            else:
                main = A.pool_ce(lp, labels, args.smoothing)
                kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
                main = main + args.beta * kl
                parts = (main.detach(), kl.detach(), main.detach(), (lp.argmax(-1) == labels).float().mean())
            sib = A.sibling_margin(lp, labels, args.margin)
            con = A.hierarchical_supcon(text_z, labels, args.temperature, args.family_pos)
            loss = main + args.lambda_sib * sib + args.lambda_con * con
            return loss, *parts, sib.detach(), con.detach()

        factor = O.lr_factor(state['batch'], len(batches), args.warmup)
        for group_ in opt.param_groups:
            group_['lr'] = group_['base_lr'] * factor
        try:
            parts = train_step(model, tok, ids, win_y, win_w, win_y, indices,
                               rt.device, args, state['micro'], torch.zeros(len(classes), device=rt.device),
                               loss_fn, rt)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            if state['micro'] == 1:
                raise
            state['micro'] = max(1, state['micro'] // 2)
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if rt.any(not torch.isfinite(grad_norm).item()):
            raise FloatingPointError('Non-finite v3 gradient; no optimizer update/export performed')
        opt.step(); opt.zero_grad(set_to_none=True)
        state['optimizer_steps'] += 1; state['batch'] += 1
        if state['optimizer_steps'] % args.ckpt_every == 0:
            save()
        log(dict(batch=state['batch'], kind=kind, total_batches=len(batches), lr_factor=round(factor, 5), ce=parts[0],
                 kl=parts[1], nll=parts[2], accuracy=parts[3], sibling=parts[4], supcon=parts[5], grad_norm=float(grad_norm)))
    save()
    validation_ids = [store.select(int(r), first=True) for r in val]
    while True:
        try:
            validation_logits = V2T.head_logits(model, tok, validation_ids, rt, state['micro'],
                                                lambda: stop['flag'] or time.time() >= deadline)
            pv = validation_logits[:, :len(classes)].softmax(-1).cpu().numpy()
            break
        except SystemExit:
            save(); raise
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if state['micro'] == 1:
                raise
            state['micro'] = max(1, state['micro'] // 2)
    if rt.primary:
        np.save(run / 'val_probs_final.npy', pv)
        np.save(run / 'val_rows.npy', val)
        S.export_public(model, run / 'trainables.safetensors')
        W.interpolate(public_state, run / 'trainables.safetensors', W.DEPLOY_ALPHA, run / 'trainables_wise050.safetensors')
        artifacts = {n: sha(run / n) for n in
                     ['trainables.safetensors', 'trainables_wise050.safetensors', 'val_probs_final.npy',
                      'val_rows.npy', 'reference.npz', 'reference.json']}
        done = dict(name='11/12-v4 A1 HSC-Native', status='TRAINED', args={**vars(args),
                    **{k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}},
                    classes=classes, world_size=rt.world, seed=args.seed, method=METHOD,
                    objective='v3 objective + sibling-structured pool replay + hierarchical supcon + sibling margin',
                    official_cells=len(official_cells_), pool_cells=len(replay), initialization='pinned public native fold state',
                    public_sha256=source_hashes, policy_calibration=calibration, calibration_mode='frozen_public',
                    wise_alpha=W.DEPLOY_ALPHA, weight_scale=weight_scale,
                    optimizer_steps=state['optimizer_steps'], batches=len(batches), resumes=state['resumes'],
                    protocol=protocol, protocol_sha256=digest, artifact_sha256=artifacts,
                    export_sha256=artifacts['trainables.safetensors'], test_run=False)
        temp = done_path.with_suffix('.tmp'); temp.write_text(json.dumps(done, indent=2) + '\n')
        os.replace(temp, done_path); checkpoint.unlink(missing_ok=True)
    rt.barrier()


def main(argv=None, model_factory=S.build_model, verifier=V2T.verify_public):
    args = parse(argv); rt = Runtime(args.device)
    lock = None
    try:
        if rt.device.type == 'cuda' and rt.world != 4:
            raise ValueError('Real v3 training requires exactly four cooperating torchrun ranks')
        error = None
        if rt.primary:
            try:
                directory = args.out / f'fold{args.fold}'
                directory.mkdir(parents=True, exist_ok=True)
                lock = (directory / '.train.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception as exc:
                error = exc
        _check_failure(error, rt)
        _run(args, rt, model_factory, verifier)
    finally:
        if lock is not None:
            lock.close()
        rt.close()


if __name__ == '__main__':
    main()
