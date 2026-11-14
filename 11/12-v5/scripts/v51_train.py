"""One GPU per 8B expert: external pretrain, then official prompt folds, immediately GPU score.
Fixed last checkpoint after 1 external / 2 official epochs. No validation checkpoint search.
Pair exclusions supply honest level-two SAGE training; they are short official-only continuations.
"""
import argparse
import datetime as dt
import itertools
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v51_common import BASE, ROOT, RUN, GROUPS, SEED, dump, sha, row_folds, save_npz, load_verified
import v51_expert as E


def excluded_sets(smoke=False):
    return [(0,)] if smoke else [*( (k,) for k in range(5)), *itertools.combinations(range(5), 2)]


def split_name(excluded):
    return 'external' if not excluded else 'exclude_' + '_'.join(map(str, excluded))


def train_stage(model, tok, texts, labels, folder, identity, deadline, epochs, lr, max_steps=0):
    import torch
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'trained.json').exists():
        rec = json.loads((folder / 'trained.json').read_text())
        if rec['identity'] != identity or sha(folder / 'final.safetensors') != rec['checkpoint_sha256']:
            raise ValueError('Completed training identity/checkpoint changed')
        E.load_state(model, folder / 'final.safetensors'); return
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('head.')], 'lr': lr},
                            {'params': list(model.head.parameters()), 'lr': lr * 5}], weight_decay=.01)
    micro, accum = 2, 16
    order_len = len(E.balanced_order(labels, np.random.default_rng(SEED)))
    per_epoch = math.ceil(order_len / (micro * accum)); total = per_epoch * epochs
    start = 0
    if (folder / 'resume.pt').exists():
        ck = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
        if ck['identity'] != identity:
            raise ValueError('Resume protocol changed; use a separate run')
        model.load_state_dict(ck['model'], strict=False); opt.load_state_dict(ck['optimizer']); start = ck['step']
        torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state(ck['cuda_rng'])
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def checkpoint(step):
        temp = folder / 'resume.tmp.pt'
        torch.save(dict(identity=identity, model=E.state(model), optimizer=opt.state_dict(), step=step,
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temp)
        temp.replace(folder / 'resume.pt')
        dump(folder / 'status.json', dict(status='PAUSED', optimizer_steps=step, total_steps=total))

    model.train()
    for ep in range(epochs):
        order = E.balanced_order(labels, np.random.default_rng([SEED, ep]))
        for s in range(per_epoch):
            step = ep * per_epoch + s
            if step < start:
                continue
            if stop['flag'] or time.time() >= deadline - 180:
                checkpoint(step); raise SystemExit(75)
            batch = order[s * micro * accum:(s + 1) * micro * accum]
            opt.zero_grad(set_to_none=True)
            factor = min(1., (step + 1) / max(1, int(total * .03))) * .5 * (1 + math.cos(math.pi * step / max(1, total)))
            opt.param_groups[0]['lr'] = lr * factor; opt.param_groups[1]['lr'] = lr * 5 * factor
            loss_value = 0.
            for pos in range(0, len(batch), micro):
                rows = batch[pos:pos + micro]
                logits = model(**E.encode(tok, [texts[i] for i in rows]))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels[rows], device='cuda'), reduction='sum') / len(batch)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite expert loss; checkpoint not marked complete')
                loss.backward(); loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
            if step % 20 == 0:
                print(json.dumps(dict(stage=str(folder), step=step + 1, total=total, ce=loss_value)), flush=True)
            if (step + 1) % 100 == 0:
                checkpoint(step + 1)
            if max_steps and step + 1 >= max_steps:
                total = step + 1; break
        if max_steps:
            break
    temp = folder / 'final.tmp.safetensors'; save_file(E.state(model), str(temp)); temp.replace(folder / 'final.safetensors')
    dump(folder / 'trained.json', dict(status='TRAINED', identity=identity, optimizer_steps=total,
         checkpoint_sha256=sha(folder / 'final.safetensors'), selection='fixed_last_epoch', smoke=bool(max_steps)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=list(GROUPS), required=True)
    p.add_argument('--base', type=Path, default=BASE / 'base_staged')
    p.add_argument('--external', type=Path, default=ROOT / 'cache/external_v51.parquet')
    p.add_argument('--out', type=Path, default=RUN / 'experts')
    p.add_argument('--deadline', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); deadline_dt = dt.datetime.fromisoformat(a.deadline)
    if deadline_dt.tzinfo is None:
        p.error('Timezone required in deadline')
    deadline = deadline_dt.timestamp()
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Each expert must see exactly one authorized GPU')
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ext_record = load_verified(a.external); ext = pd.read_parquet(a.external)
    official = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    fold = row_folds(official, json.loads((BASE / 'data/folds.json').read_text()))
    classes = GROUPS[a.group]
    ext = ext[ext.model.isin(classes) & ext.split.eq('train')]
    if a.smoke:
        ext = ext.groupby('model', group_keys=False).head(4)
    if set(ext.model) != set(classes):
        raise ValueError('Missing external expert class')
    # Hash actual base files, not an arbitrary path. The pinned asset installer checks full shard hashes.
    from v51_collect import verify_assets
    assets = verify_assets()
    if a.base.resolve() != (BASE / 'base_staged').resolve():
        if any(sha(a.base / name) != digest for name, digest in assets['base'].items()):
            raise ValueError('--base differs from the pinned staged base')
    identity = dict(method='v51_discriminative_8b', group=a.group, seed=SEED,
                    external_sha256=ext_record['payload_sha256'], official_sha256=sha(BASE / 'data/original/train.csv'),
                    folds_sha256=sha(BASE / 'data/folds.json'), base=assets['base'], code_sha256=sha(__file__),
                    expert_sha256=sha(E.__file__), plan_sha256=sha(ROOT / 'artifacts/v51_plan.json'), smoke=a.smoke)
    tok, model = E.make_model(a.base, len(classes)); folder = a.out / a.group
    external_dir = folder / 'external'
    train_stage(model, tok, ext.text.tolist(), np.array([classes.index(c) for c in ext.model]), external_dir,
                dict(identity, excluded=[]), deadline, 1, 1e-4, 2 if a.smoke else 0)
    for excluded in excluded_sets(a.smoke):
        run = folder / split_name(excluded)
        E.load_state(model, external_dir / 'final.safetensors')
        tr = np.flatnonzero(~np.isin(fold, excluded) & official.model.isin(classes).values)
        te = np.flatnonzero(np.isin(fold, excluded))  # Score ALL labels for false-route diagnostics.
        if a.smoke:
            tr = np.concatenate([tr[official.model.values[tr] == c][:4] for c in classes]); te = te[:16]
        train_stage(model, tok, official.text.iloc[tr].tolist(), np.array([classes.index(c) for c in official.model.iloc[tr]]),
                    run, dict(identity, excluded=list(excluded)), deadline, 2, 2e-5, 2 if a.smoke else 0)
        # Immediate GPU evaluation of the frozen final checkpoint; no CPU fallback.
        if (run / 'complete.json').exists():
            rec = load_verified(run / 'scores.npz')
            if rec['checkpoint_sha256'] != sha(run / 'final.safetensors'):
                raise ValueError('Completed scores use another checkpoint')
            continue
        if time.time() >= deadline - 180:
            dump(run / 'status.json', dict(status='TRAINED', pending='GPU validation'))
            raise SystemExit(75)
        lp = E.score(model, tok, official.text.iloc[te].tolist())
        save_npz(run / 'scores.npz', rows=te, ids=official.id.iloc[te].astype(str).values, lp=lp, classes=np.array(classes))
        rec = dict(status='VALIDATED', hidden_test_evaluated=False, excluded=list(excluded),
                   payload_sha256=sha(run / 'scores.npz'), checkpoint_sha256=sha(run / 'final.safetensors'),
                   identity=identity, rows=len(te), smoke=a.smoke)
        dump(run / 'scores.json', rec); dump(run / 'complete.json', rec)


if __name__ == '__main__':
    main()
