"""Four cooperating GPUs, seed 42; window-aligned prompt-pair learning."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

from shared import BASE, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--folds', default='0,1', help='Two prompt folds, one fixed seed; full protocol: 0,1,2,3')
    p.add_argument('--pool-minutes', type=float, default=90)
    p.add_argument('--finish-minutes', type=float, default=45)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--sampling', choices=['weighted', 'first'], default='weighted')
    p.add_argument('--lambda-pair', type=float, default=.2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(',')
    folds = [int(k) for k in a.folds.split(',')]
    if len(devices) != 4 or len(set(devices)) != 4:
        p.error('Four distinct GPUs are required')
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= {0, 1, 2, 3}:
        p.error('Folds must be distinct members of 0,1,2,3')
    if a.pool_minutes < 10 or a.finish_minutes <= 0 or a.micro_batch < 1 or a.lambda_pair < 0:
        p.error('Invalid time budget, micro batch or pair weight')
    if not a.smoke and (a.sampling != 'weighted' or a.lambda_pair != .2):
        run_name = f'ablation_{a.sampling}_pair{a.lambda_pair:g}'
    else:
        run_name = 'runs_smoke' if a.smoke else 'runs'
    runs = ROOT / run_name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS='8')

    def execute(command):
        print('COMMAND', json.dumps(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not a.dry_run:
        from assets import originals, augmentation
        originals(); augmentation()
        if not (BASE / 'vendor/native_q8.py').exists() or not (BASE / 'base_staged/tokenizer.json').exists():
            raise SystemExit('Run python ../11/scripts/assets.py all first')
    execute([sys.executable, str(ROOT / 'scripts/windows.py')])
    for k in (folds[:1] if a.smoke else folds):
        now = dt.datetime.now(dt.timezone.utc)
        pool_end = now + dt.timedelta(minutes=a.pool_minutes)
        finish_end = pool_end + dt.timedelta(minutes=a.finish_minutes)
        execute([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 str(ROOT / 'scripts/train.py'), '--data', str(BASE / 'data/original'),
                 '--model', str(BASE / 'base_staged'), '--folds', str(BASE / 'data/folds.json'),
                 '--fold', str(k), '--out', str(runs),
                 '--stream', str(BASE / 'data/stream_8b_cap9500.parquet'),
                 '--token-cache', str(ROOT / 'cache/tokens'), '--sampling', a.sampling,
                 '--lambda-pair', str(a.lambda_pair), '--micro-batch', str(a.micro_batch),
                 '--seed', '42', '--train-end', pool_end.isoformat(), '--stage2-end', finish_end.isoformat()]
                + (['--max-steps', '3', '--max-steps2', '2'] if a.smoke else []))
        execute([sys.executable, str(ROOT / 'scripts/evaluate.py'), '--run', str(runs / f'fold{k}'),
                 '--fold', str(k), '--gpus', '4'] + (['--smoke', '48'] if a.smoke else []))
    if not a.smoke:
        execute([sys.executable, str(ROOT / 'scripts/decide.py'), '--runs', str(runs), '--out', str(runs / 'decision.json')])


if __name__ == '__main__':
    main()
