"""v6 launcher, four cooperating GPUs; rerunning the same command resumes and reuses verified finished stages.
  CPU   data_v6 (Open Pool v15 stream) -> token cache (11/12 windows.py)
  GPU   pretrain (shared) -> per fold: finetune x seeds -> soup -> 11/12 evaluate.py (public native runtime,
        parity, inner-row calibration)
  CPU   decide_v6 (native-alone gate; decision.json)
Exit code 75 = a time window ended with a checkpoint saved."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from shared import BASE, ROOT, V1
import data_v6

PAUSED = 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--minutes', type=float, default=600, help='Wall-clock budget for this invocation (saves and pauses)')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--presentations', type=int, default=400000)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4:
        p.error('Four distinct GPU IDs are required')
    root = a.out.resolve() if a.out else ROOT / ('runs_v6_smoke' if a.smoke else 'runs_v6')
    seeds = a.seeds.split(',')[:2] if a.smoke else a.seeds.split(',')
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=a.minutes)).isoformat()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='true')
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', ROOT / 'scripts/train_v6.py']
    common = ['--out', root, '--deadline', deadline, '--micro-batch', a.micro_batch]

    def execute(command, cpu=False):
        command = [str(c) for c in command]
        print('COMMAND', json.dumps(command), flush=True)
        if a.dry_run:
            return
        code = subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES='') if cpu else env).returncode
        if code == PAUSED:
            print(json.dumps(dict(status='PAUSED', note='window ended; checkpoint saved; rerun the same command')), flush=True)
            raise SystemExit(PAUSED)
        if code:
            raise SystemExit(f'Step failed with exit code {code}: {command[:6]}')

    if not a.dry_run:
        from assets import originals
        originals()
        for need in (BASE / 'vendor/native_q8.py', BASE / 'base_staged/tokenizer.json', BASE / 'external/open_pool_v15/ntl_open_pool.parquet'):
            if not need.exists():
                raise SystemExit(f'Missing asset {need}; see README (assets.py all + Open Pool v15 download)')
    if not data_v6.DEFAULT.exists():
        execute([sys.executable, ROOT / 'scripts/data_v6.py'], cpu=True)
    cache = ROOT / 'cache/tokens_v6'
    if not (cache / 'manifest.json').exists():
        execute([sys.executable, V1 / 'scripts/windows.py', '--data', BASE / 'data/original', '--stream', data_v6.DEFAULT,
                 '--model', BASE / 'base_staged', '--out', cache], cpu=True)
    smoke_pre = ['--presentations', '512', '--max-steps', '4'] if a.smoke else ['--presentations', a.presentations]
    execute(torchrun + ['--stage', 'pretrain'] + common + smoke_pre)
    for k in ([0] if a.smoke else range(5)):
        for s in seeds:
            execute(torchrun + ['--stage', 'finetune', '--fold', k, '--seed', s] + common + (['--max-steps', '2'] if a.smoke else []))
        if a.dry_run or not (root / f'fold{k}/done.json').exists():
            execute(torchrun + ['--stage', 'soup', '--fold', k, '--seeds', ','.join(seeds)] + common)
        if a.smoke or a.dry_run or not (root / f'fold{k}/native_eval.json').exists():
            execute([sys.executable, V1 / 'scripts/evaluate.py', '--run', root / f'fold{k}', '--fold', k, '--gpus', '4']
                    + (['--smoke', '200'] if a.smoke else []))
    if not a.smoke:
        execute([sys.executable, ROOT / 'scripts/decide_v6.py', '--runs', root], cpu=True)
    print(json.dumps(dict(status='SMOKE_DONE' if a.smoke else 'DONE', root=str(root))), flush=True)


if __name__ == '__main__':
    main()
