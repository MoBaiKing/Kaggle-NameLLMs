"""Four-rank enumerated GRPO v2 with frozen multi-model soft voting.

All native windows participate, both public heads are optimized, and public
calibration is frozen. No fresh SFT, critic, language generation or test labels.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
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
from safetensors.torch import load_file

from shared import BASE, ROOT, V1
import train as S
from distributed import Runtime, _check_failure, _gather_rows, train_step
from grpo import aggregate_windows, native_log_probs
from grpo_v2 import ensemble_log_probs, enumerate_group, objective
from public_parent import load_parent
from helpers import atomic_save, official_cells
from windows import TokenStore, write_cache, sha

NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'
META = BASE / 'external/native_q8_meta/ntl-native-q8-multidepth-fivefold-0679'


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=BASE / 'data/original')
    p.add_argument('--model', type=Path, default=BASE / 'base_staged')
    p.add_argument('--folds', type=Path, default=BASE / 'data/folds.json')
    p.add_argument('--native-dir', type=Path, default=NATIVE)
    p.add_argument('--native-meta', type=Path, default=META)
    p.add_argument('--token-cache', type=Path, default=V1 / 'cache/grpo_tokens')
    p.add_argument('--reference-source', type=Path, default=V1 / 'runs_grpo',
                   help='Reuse a verified completed v1 public reference when available')
    p.add_argument('--out', type=Path, default=ROOT / 'runs_grpo')
    p.add_argument('--fold', type=int, required=True, choices=range(5))
    p.add_argument('--deadline', required=True, help='ISO time; checkpoint and exit 75 at deadline')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=16, help='GLOBAL text batch, shared by all four ranks')
    p.add_argument('--micro-batch', type=int, default=2, help='Windows per rank, gradient cached')
    p.add_argument('--updates', type=int, default=2, help='Clipped optimizer updates per frozen rollout')
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--beta', type=float, default=.02)
    p.add_argument('--ce-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42, choices=[42])
    p.add_argument('--max-batches', type=int, default=0, help='Smoke test only; use a separate output')
    p.add_argument('--ckpt-every', type=int, default=20)
    p.add_argument('--mem-cap-gb', type=float, default=28.5)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU only for synthetic tests')
    a = p.parse_args(argv)
    if min(a.epochs, a.batch, a.micro_batch, a.updates, a.ckpt_every) < 1:
        p.error('Invalid batch/epoch/update count')
    if not 0 < a.epsilon < 1 or min(a.beta, a.ce_weight) < 0 or a.lr <= 0 or a.max_batches < 0 or a.mem_cap_gb <= 0:
        p.error('Invalid GRPO coefficient, learning rate or budget')
    if datetime.fromisoformat(a.deadline).tzinfo is None:
        p.error('--deadline must contain a timezone')
    return a


def token_identity(args):
    return dict(format='grpo-official-native-windows-v1', train=sha(args.data / 'train.csv'),
                tokenizer={n: sha(args.model / n) for n in
                           ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'merges.txt', 'vocab.json']},
                transformers=importlib.metadata.version('transformers'),
                tokenizers=importlib.metadata.version('tokenizers'))


def verify_public(args, frame, fingerprint):
    """Fail before loading an 8B model if any upstream identity/split is wrong."""
    spec = json.loads((BASE / 'config/assets.json').read_text())
    manifest_file = args.native_dir / 'manifest.json'
    if sha(manifest_file) != spec['native_assets']['manifest_sha256']:
        raise ValueError('Public native manifest changed')
    manifest = json.loads(manifest_file.read_text())
    if manifest['classes'] != sorted(frame.model.unique()) or manifest['revision'] != spec['model']['revision']:
        raise ValueError('Public classes/backbone revision differ')
    if fingerprint['train'] != spec['original_data']['train.csv']['sha256'] or sha(args.folds) != spec['folds_sha256']:
        raise ValueError('Official data/folds changed')
    for name, value in fingerprint['tokenizer'].items():
        if value != manifest['file_hashes'][f'base/{name}']:
            raise ValueError(f'Public tokenizer/config changed: {name}')
    paths = dict(state=args.native_dir / f'fold{args.fold}/trainables.safetensors',
                 calibration=args.native_dir / f'calibration/fold{args.fold}.json',
                 split=args.native_meta / f'fold{args.fold}/split.json')
    relatives = dict(state=f'fold{args.fold}/trainables.safetensors',
                     calibration=f'calibration/fold{args.fold}.json', split=f'fold{args.fold}/split.json')
    for key, path in paths.items():
        if sha(path) != manifest['file_hashes'][relatives[key]]:
            raise ValueError(f'Public fold asset changed: {path}')
    split = json.loads(paths['split'].read_text())
    fit = np.asarray(split['training_rows'], dtype=int)
    val = np.asarray(split['validation_rows'], dtype=int)
    folds = json.loads(args.folds.read_text())
    if (not split['official_oof_eligible'] or set(val) != set(folds[str(args.fold)])
            or len(set(fit)) != len(fit) or len(set(val)) != len(val)
            or set(fit) & set(val) or set(fit) | set(val) != set(range(len(frame)))):
        raise ValueError('Public training/validation split is invalid')
    if set(frame.prompt_id.iloc[fit]) & set(frame.prompt_id.iloc[val]):
        raise ValueError('Public prompt split leaks across folds')
    calibration = json.loads(paths['calibration'].read_text())['models']['qwen8']
    if (calibration['outer_labels_used'] or not calibration['centered_head_trained']
            or min(calibration['plain_scale'], calibration['centered_scale']) <= 0
            or not 0 <= calibration['selected_centered_weight'] <= .5):
        raise ValueError('Invalid public calibration')
    return fit, val, calibration, {k: sha(v) for k, v in paths.items()}


def load_public(model, path):
    """Exact inverse of train.export_public; no randomly initialized head survives."""
    public = load_file(str(path), device='cpu')
    expected, mapped = {}, {}
    for name, module in model.backbone.named_modules():
        if isinstance(module, S.TaskLoRA):
            prefix = f'encoder.backbone.base_model.model.{name}'
            expected[f'{prefix}.lora_A.default.weight'] = f'backbone.{name}.a.weight'
            expected[f'{prefix}.lora_B.default.weight'] = f'backbone.{name}.b.weight'
    for prefix in ('norms', 'plain', 'centered'):
        for name, _ in getattr(model, prefix).named_parameters():
            expected[f'{prefix}.{name}'] = f'{prefix}.{name}'
    if set(public) != set(expected):
        raise ValueError('Public state layout differs from native architecture')
    current = model.state_dict()
    for key, target in expected.items():
        if public[key].shape != current[target].shape or not torch.isfinite(public[key]).all():
            raise ValueError(f'Invalid public tensor: {key}')
        mapped[target] = public[key]
    model.load_state_dict(mapped, strict=False)
    model.proj.requires_grad_(False)
    # Keep gradient checkpointing active (HF disables it in eval mode), while
    # preventing random masks from corrupting old/current policy ratios.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.
    model.train()


@torch.no_grad()
def head_logits(model, tok, sequences, rt, micro, halt=None):
    """Bounded no-grad forwards, one global row order, unequal/empty ranks allowed."""
    local = sequences[rt.rank::rt.world]
    outputs = []
    width = (len(sequences) + rt.world - 1) // rt.world
    for start in range(0, width, micro):
        if halt is not None and rt.any(halt()):
            raise SystemExit(75)
        error = None
        try:
            if start < len(local):
                batch = tok.pad({'input_ids': local[start:start + micro]}, padding=True, return_tensors='pt').to(rt.device)
                features = model.features(batch['input_ids'], batch['attention_mask'])
                outputs.append(torch.cat([model.plain(features).float(), model.centered(features).float()], -1))
        except Exception as exc:
            error = exc
        _check_failure(error, rt)
    value = torch.cat(outputs) if outputs else torch.empty((0, 2 * model.plain.out_features), device=rt.device)
    return _gather_rows(value, len(sequences), rt)


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
    identity = token_identity(args)
    fit, val, calibration, source_hashes = verifier(args, frame, identity)
    parent, parent_record = load_parent(frame.prompt_id.values, classes)
    code_files = {n: Path(__file__).with_name(n) for n in
                  ['grpo_train.py', 'grpo_v2.py', 'public_parent.py', 'runtime_vote.py', 'shared.py']}
    code_files.update({f'v1/{n}': V1 / 'scripts' / n for n in
                       ['grpo.py', 'train.py', 'distributed.py', 'windows.py']})
    code_files['helpers.py'] = Path(official_cells.__code__.co_filename)
    protocol = dict(method='public-native-enumerated-GRPO-v2', inputs=identity,
                    voting=dict(parent_record=parent_record, proxy_native_share=.40,
                                full_ensemble_share=.45, explicit_voters_share=.15),
                    folds=sha(args.folds), public=source_hashes, calibration=calibration,
                    execution=dict(torch=torch.__version__, world_size=rt.world,
                                   device=rt.device.type, attention='sdpa'),
                    arguments={k: v for k, v in vars(args).items() if k not in
                               {'data', 'model', 'folds', 'native_dir', 'native_meta', 'token_cache',
                                'out', 'deadline', 'device', 'ckpt_every', 'mem_cap_gb', 'reference_source'}},
                    code={n: sha(path) for n, path in code_files.items()})
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    done_path, checkpoint = run / 'done.json', run / 'checkpoint.pt'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done['protocol_sha256'] != digest or any(sha(run / n) != h for n, h in done['artifact_sha256'].items()):
            raise ValueError('Completed GRPO run inputs or artifacts changed')
        log(dict(reused=True, status='TRAINED', fold=args.fold))
        return
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if rt.scalar(time.time()) >= deadline:
        raise SystemExit(75)
    torch.manual_seed(args.seed)
    architecture = SimpleNamespace(model=str(args.model), lora_drop=0., feat_drop=0., proj_dim=256)
    tok, model = model_factory(architecture, len(classes), rt.device)
    load_public(model, args.native_dir / f'fold{args.fold}/trainables.safetensors')
    cache_error = None
    if rt.primary:
        try:
            write_cache(frame.text.values, tok, args.token_cache, identity)
        except Exception as exc:
            cache_error = exc
    _check_failure(cache_error, rt)
    store = TokenStore(args.token_cache)
    if store.manifest['identity'] != identity or len(store) != len(frame):
        raise ValueError('GRPO token cache inputs differ')
    sequences, starts, lengths, window_weights = [], {}, {}, []
    for row in fit:
        seqs, weights = store.all_windows(int(row))
        starts[int(row)], lengths[int(row)] = len(sequences), len(seqs)
        sequences.extend(seqs); window_weights.extend(weights)
    window_weights = np.asarray(window_weights, np.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.)
    batches = [cell for epoch in range(args.epochs) for cell in
               official_cells(frame.prompt_id.values, y, fit, 4, 4, args.batch,
                              np.random.default_rng([args.seed, epoch, 320]))]
    if args.max_batches:
        batches = batches[:args.max_batches]
    state = dict(batch=0, update=0, optimizer_steps=0, resumes=0, micro=args.micro_batch, pending=None)
    saved = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest or saved['world_size'] != rt.world:
            raise ValueError('Resume needs the same GRPO protocol and rank count')
        state = saved['state']; state['resumes'] += 1
    ref_path = run / 'reference.npz'
    ref_record = run / 'reference.json'
    # Reference outputs depend on the frozen public model/windows, not the RL
    # estimator. Reuse one verified v1 file via symlink; never load its student.
    candidate = args.reference_source / f'fold{args.fold}'
    if not ref_path.exists() and saved is None and (candidate / 'done.json').exists():
        source = json.loads((candidate / 'done.json').read_text())
        source_protocol = source['protocol']
        expected = hashlib.sha256(json.dumps(source_protocol, sort_keys=True).encode()).hexdigest()
        compatible = (source_protocol.get('method') == 'public-native-categorical-GRPO-v1'
                      and source_protocol.get('inputs') == identity
                      and source_protocol.get('public') == source_hashes
                      and source_protocol.get('folds') == protocol['folds']
                      and source_protocol.get('execution') == protocol['execution'])
        if compatible:
            if source['protocol_sha256'] != expected:
                raise ValueError('Reusable v1 reference protocol changed')
            for name, fingerprint in source_protocol['code'].items():
                if sha(V1 / 'scripts' / name) != fingerprint:
                    raise ValueError('Reusable v1 reference code changed')
            reference_hash = sha(candidate / 'reference.npz')
            if reference_hash != source['artifact_sha256']['reference.npz']:
                raise ValueError('Reusable v1 reference bytes changed')
            with np.load(candidate / 'reference.npz', allow_pickle=False) as z:
                if str(z['protocol_sha256'].item()) != expected:
                    raise ValueError('Reusable v1 reference payload changed')
            if rt.primary:
                ref_path.symlink_to((candidate / 'reference.npz').resolve())
                temp = ref_record.with_suffix('.tmp')
                temp.write_text(json.dumps(dict(protocol_sha256=digest, sha256=reference_hash,
                    payload_protocol_sha256=expected, reused_v1_done_sha256=sha(candidate / 'done.json'))) + '\n')
                os.replace(temp, ref_record)
            rt.barrier()
            log(dict(reused_public_reference=True, source=str(candidate)))
    if ref_path.exists():
        if not ref_record.exists():
            raise ValueError('Reference cache is missing its integrity record')
        reference_record = json.loads(ref_record.read_text())
        if reference_record['protocol_sha256'] != digest or reference_record['sha256'] != sha(ref_path):
            raise ValueError('Frozen public reference cache changed')
        if saved is not None and sha(ref_path) != saved['reference_sha256']:
            raise ValueError('Reference differs from the resumed checkpoint')
        with np.load(ref_path, allow_pickle=False) as z:
            if str(z['protocol_sha256'].item()) != reference_record.get('payload_protocol_sha256', digest):
                raise ValueError('Frozen reference protocol changed')
            reference = torch.tensor(z['logits'], device=rt.device)
    else:
        if saved is not None:
            raise ValueError('Resume is missing its frozen public reference')
        while True:
            try:
                reference = head_logits(model, tok, sequences, rt, state['micro'],
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

    def save():
        rng = rt.checkpoint_rng()
        if rt.primary:
            atomic_save(dict(model=S.trainable_state(model), opt=opt.state_dict(), state=state,
                             rng_states=rng, world_size=rt.world, protocol_sha256=digest,
                             reference_sha256=reference_sha), checkpoint)
        rt.barrier()

    save()
    # y/weights/groups are supplied to the shared gradient-cache interface; the
    # objective below uses the TEXT rows after differentiable window aggregation.
    win_y = np.zeros(len(sequences), np.int64)
    win_w = np.ones(len(sequences), np.float32)
    ids = {i: seq for i, seq in enumerate(sequences)}
    while state['batch'] < len(batches):
        if rt.any(stop['flag']) or rt.scalar(time.time()) >= deadline:
            save(); raise SystemExit(75)
        cell = batches[state['batch']]
        indices, owners = [], []
        for owner, row in enumerate(cell):
            indices.extend(range(starts[row], starts[row] + lengths[row]))
            owners.extend([owner] * lengths[row])
        owners = torch.tensor(owners, device=rt.device)
        weights = torch.tensor(window_weights[indices], device=rt.device)
        labels = torch.tensor(y[cell], device=rt.device)
        group = torch.tensor(groups[cell], device=rt.device)
        balanced = torch.tensor(class_weights[y[cell]], dtype=torch.float32, device=rt.device)
        parent_lp = torch.tensor(parent[cell], device=rt.device)
        reduce = lambda logits: aggregate_windows(logits, owners, weights, len(cell))
        rp, rc = reference[indices].chunk(2, -1)
        ref_lp = native_log_probs(reduce(rp), reduce(rc), ref_bias, group, calibration).detach()
        if state['pending'] is None:
            while True:
                try:
                    old = head_logits(model, tok, [sequences[i] for i in indices], rt, state['micro'],
                                      lambda: stop['flag'] or time.time() >= deadline)
                    break
                except SystemExit:
                    save(); raise
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if state['micro'] == 1:
                        raise
                    state['micro'] = max(1, state['micro'] // 2)
            op, oc = old.chunk(2, -1)
            old_lp = native_log_probs(reduce(op), reduce(oc), model.centered.bias.detach(), group, calibration)
            state['pending'] = enumerate_group(ensemble_log_probs(old_lp, parent_lp), labels)

        def loss_fn(plain, centered, z, lab, w, gid, bias, prior, unused):
            lp = native_log_probs(reduce(plain), reduce(centered), bias, group, calibration)
            policy = ensemble_log_probs(lp, parent_lp)
            loss, parts = objective(policy, lp, ref_lp, labels, balanced, state['pending'],
                                    args.epsilon, args.beta, args.ce_weight)
            return (loss + z.sum() * 0), *parts

        while state['update'] < args.updates:
            if rt.any(stop['flag']) or rt.scalar(time.time()) >= deadline:
                save(); raise SystemExit(75)
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
                raise FloatingPointError('Non-finite GRPO gradient; no optimizer update/export performed')
            opt.step(); opt.zero_grad(set_to_none=True)
            state['optimizer_steps'] += 1; state['update'] += 1
            if state['optimizer_steps'] % args.ckpt_every == 0:
                save()
            log(dict(batch=state['batch'], update=state['update'], total_batches=len(batches),
                     pg=parts[0], kl=parts[1], ce=parts[2], degenerate_policy_fraction=parts[3], clipped_old_mass=parts[4], expected_correct_reward=parts[5]))
        state.update(batch=state['batch'] + 1, update=0, pending=None)
    save()
    validation_ids = [store.select(int(r), first=True) for r in val]
    while True:
        try:
            validation_logits = head_logits(model, tok, validation_ids, rt, state['micro'],
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
        np.save(run / 'inner_rows.npy', np.array([], dtype=np.int64))
        S.export_public(model, run / 'trainables.safetensors')
        artifacts = {n: sha(run / n) for n in
                     ['trainables.safetensors', 'val_probs_final.npy', 'val_rows.npy', 'inner_rows.npy',
                      'reference.npz', 'reference.json']}
        done = dict(name='11/12-v2 enumerated GRPO multi-model vote', status='TRAINED', args={**vars(args),
                    **{k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}},
                    classes=classes, world_size=rt.world, seed=args.seed,
                    initialization='pinned public native fold state', public_sha256=source_hashes,
                    policy_calibration=calibration, calibration_mode='frozen_public',
                    method='public_native_grpo_v2', voting=protocol['voting'], estimator='enumerated_full_support',
                    optimizer_steps=state['optimizer_steps'], batches=len(batches), resumes=state['resumes'],
                    protocol=protocol, protocol_sha256=digest, artifact_sha256=artifacts,
                    export_sha256=artifacts['trainables.safetensors'], test_run=False)
        temp = done_path.with_suffix('.tmp'); temp.write_text(json.dumps(done, indent=2) + '\n')
        os.replace(temp, done_path); checkpoint.unlink(missing_ok=True)
    rt.barrier()


def main(argv=None, model_factory=S.build_model, verifier=verify_public):
    args = parse(argv); rt = Runtime(args.device)
    lock = None
    try:
        if rt.device.type == 'cuda' and rt.world != 4:
            raise ValueError('Real GRPO training requires exactly four cooperating torchrun ranks')
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
