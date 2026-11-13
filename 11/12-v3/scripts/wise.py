"""WiSE-FT weight-space interpolation between the public native state and a continued state (same init).

alpha = 1 is the continued state, alpha = 0 the public one. Every tensor (LoRA A/B, layer norms, both heads)
is interpolated in float64 and stored as float32 in the unchanged public key layout, so native_q8 loads it
as is. Also prepares v2 GRPO states as the Option-A fallback family without touching the v2 run directory.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from shared import BASE, ROOT
from oof_targets import sha

NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'
DEPLOY_ALPHA = .5


def interpolate(public, tuned, alpha, out):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1]')
    a, b = load_file(str(public)), load_file(str(tuned))
    if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError('Interpolated states must share the public layout')
    mixed = {k: ((1 - alpha) * a[k].double() + alpha * b[k].double()).float().contiguous() for k in sorted(a)}
    if not all(torch.isfinite(v).all() for v in mixed.values()):
        raise ValueError('Non-finite interpolated tensor')
    out = Path(out); temp = str(out) + '.tmp'
    save_file(mixed, temp); os.replace(temp, out)
    return sha(out)


def from_v2(v2_runs, out, folds):
    """Fallback family A: verified v2 states (alpha 1) plus their alpha 0.5 interpolation."""
    made = {}
    for k in folds:
        source = Path(v2_runs) / f'fold{k}'
        done = json.loads((source / 'done.json').read_text())
        if done.get('method') != 'public_native_grpo_v2' or done['args']['fold'] != k:
            raise ValueError(f'fold{k}: not a completed v2 GRPO run')
        for name, expected in done['artifact_sha256'].items():
            if name in ('trainables.safetensors', 'val_probs_final.npy', 'val_rows.npy') and sha(source / name) != expected:
                raise ValueError(f'fold{k}: v2 artifact changed: {name}')
        public = NATIVE / f'fold{k}/trainables.safetensors'
        if sha(public) != done['public_sha256']['state']:
            raise ValueError(f'fold{k}: v2 run did not start from the pinned public state')
        target = Path(out) / f'fold{k}'; target.mkdir(parents=True, exist_ok=True)
        for name in ('trainables.safetensors', 'val_probs_final.npy', 'val_rows.npy'):
            link = target / name
            if not link.exists():
                link.symlink_to((source / name).resolve())
        wise = target / 'trainables_wise050.safetensors'
        if not wise.exists():
            interpolate(public, source / 'trainables.safetensors', DEPLOY_ALPHA, wise)
        record = dict(family='v2_grpo', fold=k, source=str(source.resolve()), source_done_sha256=sha(source / 'done.json'),
                      public_sha256=sha(public), state_sha256=sha(target / 'trainables.safetensors'),
                      wise050_sha256=sha(wise), alpha=DEPLOY_ALPHA)
        (target / 'source.json').write_text(json.dumps(record, indent=1) + '\n')
        made[k] = record
    return made


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-v2', type=Path, required=True, help='11/12-v2/runs_grpo')
    p.add_argument('--out', type=Path, default=ROOT / 'runs_v3/fallback_v2')
    p.add_argument('--folds', default='0,1,2,3,4')
    a = p.parse_args()
    print(json.dumps(from_v2(a.from_v2, a.out, [int(k) for k in a.folds.split(',')]), indent=1))
