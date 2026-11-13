"""v4 launcher, four cooperating GPUs. Re-running the same command resumes; finished stages are verified and reused.
  CPU   external_v4 (prompt-clean external sets), oof_targets_v4 (anchored OOF weights)
  A1    per fold: a1_train (torchrun x4) -> a1_evaluate (4 GPUs, public native_q8 runtime, parity)
  A2    a2_train (torchrun x4, external data only) -> a2_score (torchrun x4, all 18 classes on 2,699 official rows)
  B     decide_v4 (CPU): staged evidence S0..S3 with nested CV, final SAGE bundle
Exit code 75 = a training window ended with a checkpoint; rerun the same command to resume."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from shared import BASE, ROOT
import assets_v4
import external_v4
import oof_targets_v4

PAUSED = 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--stages', default='a1,a2,b', help='Subset of a1,a2,b (in this order)')
    p.add_argument('--a1-minutes', type=float, default=150, help='Per-fold A1 window; expiry saves and pauses')
    p.add_argument('--a2-minutes', type=float, default=360, help='A2 window; expiry saves and pauses')
    p.add_argument('--a2-base', type=Path, default=assets_v4.DEFAULT)
    p.add_argument('--out', type=Path, default=None, help='Separate run root for a new fixed configuration')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or not all(x.isdigit() for x in devices):
        p.error('Four distinct GPU IDs are required')
    stages = a.stages.split(',')
    if not set(stages) <= {'a1', 'a2', 'b'}:
        p.error('Unknown stage')
    root = (a.out.resolve() if a.out else ROOT / ('runs_v4_smoke' if a.smoke else 'runs_v4'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='true')

    def execute(command, cpu=False):
        print('COMMAND', json.dumps([str(c) for c in command]), flush=True)
        if a.dry_run:
            return
        code = subprocess.run([str(c) for c in command], cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES='') if cpu else env).returncode
        if code == PAUSED:
            print(json.dumps(dict(status='PAUSED', note='window ended; checkpoint saved; rerun the same command')), flush=True)
            raise SystemExit(PAUSED)
        if code:
            raise SystemExit(f'Step failed with exit code {code}: {command[1] if len(command) > 1 else command}')

    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    if not a.dry_run:
        from assets import originals
        originals()
        if not (BASE / 'vendor/native_q8.py').exists() or not (BASE / 'base_staged/tokenizer.json').exists():
            raise SystemExit('Prepare pinned public assets with python ../11/scripts/assets.py all')
    if not external_v4.DEFAULT.exists():
        execute([sys.executable, ROOT / 'scripts/external_v4.py'], cpu=True)
    if oof_targets_v4.DEFAULT.exists():
        execute([sys.executable, ROOT / 'scripts/oof_targets_v4.py', '--check'], cpu=True)
    else:
        execute([sys.executable, ROOT / 'scripts/oof_targets_v4.py'], cpu=True)
    if 'a1' in stages:
        for k in ([0] if a.smoke else range(5)):
            deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=a.a1_minutes)
            execute(torchrun + [ROOT / 'scripts/a1_train.py', '--fold', k, '--out', root / 'a1', '--deadline', deadline.isoformat()]
                    + (['--max-batches', '4'] if a.smoke else []))
            if a.smoke or a.dry_run or not (root / f'a1/fold{k}/native_eval.json').exists():
                execute([sys.executable, ROOT / 'scripts/a1_evaluate.py', '--run', root / f'a1/fold{k}', '--fold', k, '--gpus', '4']
                        + (['--smoke', '48'] if a.smoke else []))
    if 'a2' in stages:
        if not a.dry_run:
            assets_v4.verify(a.a2_base)
        deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=a.a2_minutes)
        execute(torchrun + [ROOT / 'scripts/a2_train.py', '--base', a.a2_base, '--out', root / 'a2', '--deadline', deadline.isoformat()]
                + (['--max-steps', '3'] if a.smoke else []))
        if a.smoke or a.dry_run or not (root / 'a2/a2_scores.json').exists():
            execute(torchrun + [ROOT / 'scripts/a2_score.py', '--base', a.a2_base, '--run', root / 'a2']
                    + (['--smoke', '32'] if a.smoke else []))
    if 'b' in stages and not a.smoke:
        execute([sys.executable, ROOT / 'scripts/decide_v4.py', '--a1', root / 'a1', '--a2', root / 'a2', '--out', root / 'b'], cpu=True)
    print(json.dumps(dict(status='SMOKE_DONE' if a.smoke else 'DONE', root=str(root))), flush=True)


if __name__ == '__main__':
    main()
