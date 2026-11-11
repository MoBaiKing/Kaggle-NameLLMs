"""One fixed seed; four Ada GPUs cooperate on each fold, followed by validation."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--folds', default='0,1', help='Fast deployable subset: 0,1. Full protocol: 0,1,2,3. One model: 0.')
    p.add_argument('--pool-minutes', type=float, default=90)
    p.add_argument('--finish-minutes', type=float, default=45)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Only 3+2 training steps and public-runtime parity; no experiment result')
    a = p.parse_args()
    folds = [int(x) for x in a.folds.split(',')]
    devices = a.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4:
        p.error('Exactly four distinct GPU identifiers are required')
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {0, 1, 2, 3}:
        p.error('Folds must be distinct members of 0,1,2,3')
    if a.pool_minutes < 10 or a.finish_minutes <= 0 or a.micro_batch < 1:
        p.error('Pool budget must be >=10 minutes; finish budget and micro batch must be positive')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2')
    print(json.dumps(dict(name='11/11', seed=42, world_size=4,
                          hardware='4 x NVIDIA RTX 5000 Ada', folds=folds,
                          global_batch=16, micro_batch_per_rank=a.micro_batch)), flush=True)

    def execute(command):
        print('COMMAND', json.dumps(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not a.dry_run:
        from assets import originals, augmentation
        originals(); augmentation()
        required = [ROOT / 'base_staged/tokenizer.json', ROOT / 'vendor/native_q8.py']
        if not all(path.exists() for path in required):
            raise SystemExit('Run python scripts/assets.py all before training')
    for fold in (folds[:1] if a.smoke else folds):
        now = dt.datetime.now(dt.timezone.utc)
        pool_end = now + dt.timedelta(minutes=a.pool_minutes)
        finish_end = pool_end + dt.timedelta(minutes=a.finish_minutes)
        execute([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 str(ROOT / 'scripts/train.py'), '--data', str(ROOT / 'data/original'),
                 '--model', str(ROOT / 'base_staged'), '--folds', str(ROOT / 'data/folds.json'),
                 '--fold', str(fold), '--out', str(ROOT / ('runs_smoke' if a.smoke else 'runs')),
                 '--stream', str(ROOT / 'data/stream_8b_cap9500.parquet'),
                 '--train-end', pool_end.isoformat(), '--stage2-end', finish_end.isoformat(),
                 '--micro-batch', str(a.micro_batch), '--seed', '42'] + (['--max-steps', '3', '--max-steps2', '2'] if a.smoke else []))
        execute([sys.executable, str(ROOT / 'scripts/evaluate.py'), '--run', str(ROOT / ('runs_smoke' if a.smoke else 'runs') / f'fold{fold}'),
                 '--fold', str(fold), '--gpus', '4'] + (['--smoke', '48'] if a.smoke else []))
    if not a.smoke:
        execute([sys.executable, str(ROOT / 'scripts/decide.py')])


if __name__ == '__main__':
    main()
