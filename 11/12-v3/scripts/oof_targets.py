"""CPU only, no model is run: the frozen parent and per-row training weights from pinned clean OOF.

Why: the public fold-k native state was trained on its training rows for two epochs, so its own predictions
there are in-sample and over-confident. v2 built its ensemble policy from them, which leaves almost no signal
on the rows the deployed ensemble really gets wrong. Here the per-row difficulty comes from the author's
OUT-OF-FOLD native outputs (each row scored by the state that never saw it) fused with the label-free LoRA OOF
parent under the exact public rule (proxy_v3.deploy):

    p_oof(row) = mean over 10 test-like groupings of deploy(parent, native_oof)[row, y_row]
    weight(row) = EPS + (1 - EPS) * (1 - p_oof(row)) ** GAMMA

The parent is the three public LoRA members' OOF stack (q14v5 3 / q8v5 2 / ll8v5 1, the v2 label-free fold
mapping) sharpened by PARENT_TEMPERATURE as the public blend does. Rows of fold k only enter fold k's
VALIDATION, never its training, so fold-k labels do not reach the fold-k student through these weights
except via the other folds' states (standard stacking second-order effect).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from shared import BASE, ROOT
import proxy_v3 as X

EPS = .10
GAMMA = 1.0
SEED0 = 1000
DEFAULT = ROOT / 'cache/oof_targets.npz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def weight_of(p_true, eps=EPS, gamma=GAMMA):
    return eps + (1 - eps) * np.power(np.clip(1 - p_true, 0, 1), gamma)


def write(path, payload, record):
    """Atomic npz + json pair; record must name the official train/folds hashes it was built for."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.npz')
    np.savez(temp, **payload)
    os.replace(temp, path)
    record = dict(record, payload_sha256=sha(path), rows=int(len(payload['weight'])))
    meta = path.with_suffix('.json'); temp = meta.with_name(meta.name + '.tmp')
    temp.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    os.replace(temp, meta)
    return record


def load_verified(path, train_csv, folds_json, rows):
    """Trainer-side check: payload bytes, official data and fold identities, finite values."""
    path = Path(path); meta = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != meta['payload_sha256']:
        raise ValueError('OOF target payload changed after it was recorded')
    if meta['train_sha256'] != sha(train_csv) or meta['folds_sha256'] != sha(folds_json) or meta['rows'] != rows:
        raise ValueError('OOF targets were built for different official data or folds')
    with np.load(path, allow_pickle=False) as z:
        parent, weight = z['parent_lp'].astype(np.float64), z['weight'].astype(np.float64)
    if parent.shape != (rows, X.CLASSES) or weight.shape != (rows,):
        raise ValueError('Invalid OOF target shapes')
    if not (np.isfinite(parent).all() and np.isfinite(weight).all() and (weight > 0).all()):
        raise ValueError('Invalid OOF target values')
    return parent, weight, meta


def inputs():
    """Every file the targets depend on, by SHA256."""
    import common as C
    spec = json.loads((BASE / 'config/assets.json').read_text())
    files = {'data/original/train.csv': BASE / 'data/original/train.csv', 'data/folds.json': BASE / 'data/folds.json',
             'native/oof_reference.npz': C.NQ / 'oof_reference.npz', 'native/manifest.json': C.NW / 'manifest.json',
             'vendor/native_q8.py': BASE / 'vendor/native_q8.py', '11/11/scripts/common.py': BASE / 'scripts/common.py',
             '12-v3/scripts/oof_targets.py': Path(__file__), '12-v3/scripts/proxy_v3.py': Path(X.__file__)}
    for k in range(5):
        files[f'native/fold{k}/split.json'] = C.NQ / f'fold{k}/split.json'
        files[f'native/calibration/fold{k}.json'] = C.NW / f'calibration/fold{k}.json'
        files[f'native/fold{k}/trainables.safetensors'] = C.NW / f'fold{k}/trainables.safetensors'
    for name in ('q14v5', 'q8v5', 'll8v5'):
        files[f'data/reference/{name}.npz'] = BASE / f'data/reference/{name}.npz'
    hashes = {name: sha(path) for name, path in files.items()}
    for name in ('q14v5', 'q8v5', 'll8v5'):
        if hashes[f'data/reference/{name}.npz'] != spec['reference_data'][f'{name}.npz']['sha256']:
            raise ValueError(f'Pinned LoRA OOF changed: {name}')
    if hashes['native/manifest.json'] != spec['native_assets']['manifest_sha256']:
        raise ValueError('Pinned public native manifest changed')
    manifest = json.loads((C.NW / 'manifest.json').read_text())
    for k in range(5):
        for rel in (f'calibration/fold{k}.json', f'fold{k}/trainables.safetensors'):
            if hashes[f'native/{rel}'] != manifest['file_hashes'][rel]:
                raise ValueError(f'Pinned public native file changed: {rel}')
    return hashes


def build():
    import common as C
    hashes = inputs()
    parent = X.lsm(X.PARENT_TEMPERATURE * C.public_lora_stack())
    plain, centered, row_fold = C.public_native_oof()
    states = {k: C.public_state(k) for k in range(5)}
    biases = {k: C.public_bias(k) for k in range(5)}
    prior = X.label_prior(C.y_all)
    p_true = np.zeros(C.N)
    final = np.zeros((C.N, X.CLASSES))
    for draw in range(X.DRAWS):
        groups = X.test_like_groups(C.pid, X.GROUP_MAX, SEED0 + draw)
        native = X.native_rows(C.native_q8.calibrate, plain, centered, row_fold, biases, states, groups)
        probs = X.deploy(parent, native, prior)
        p_true += probs[np.arange(C.N), C.y_all] / X.DRAWS
        final += probs / X.DRAWS
    weight = weight_of(p_true)
    payload = dict(parent_lp=parent.astype(np.float32), p_true=p_true, weight=weight, row_fold=row_fold)
    summary = dict(public_rule_oof_ba=100 * X.ba(C.y_all, final.argmax(1)),
                   parent_only_ba=100 * X.ba(C.y_all, parent.argmax(1)),
                   mean_weight=float(weight.mean()), rows_p_true_below_half=int((p_true < .5).sum()),
                   weight_quantiles=dict(zip(['q10', 'q50', 'q90'], np.quantile(weight, [.1, .5, .9]).round(4).tolist())))
    record = dict(method='oof_targets_v3', eps=EPS, gamma=GAMMA, draws=X.DRAWS, group_max=X.GROUP_MAX,
                  seed0=SEED0, parent_temperature=X.PARENT_TEMPERATURE, native_share=X.NATIVE_SHARE,
                  sinkhorn=X.SINKHORN, target_mix=X.TARGET_MIX, inputs=hashes, summary=summary,
                  train_sha256=hashes['data/original/train.csv'], folds_sha256=hashes['data/folds.json'],
                  split='official train rows only; every native output is the author OOF of the held-out fold state')
    return payload, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--check', action='store_true', help='Rebuild in memory and require identical inputs/payload')
    a = p.parse_args()
    if a.check:
        meta = json.loads(a.out.with_suffix('.json').read_text())
        if meta['inputs'] != inputs() or sha(a.out) != meta['payload_sha256']:
            raise SystemExit('OOF targets are stale; rebuild them in a new cache path')
        print(json.dumps(dict(oof_targets='verified', **meta['summary'])))
        return
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use --check or a new path')
    payload, record = build()
    record = write(a.out, payload, record)
    print(json.dumps(dict(oof_targets=str(a.out), payload_sha256=record['payload_sha256'], **record['summary']), indent=1))


if __name__ == '__main__':
    main()
