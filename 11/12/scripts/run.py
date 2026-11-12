"""Four cooperating GPUs: public native fold -> GRPO -> native validation -> build plan."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from shared import BASE, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--folds', default='0,1,2,3,4', help='Fixed before training; all five public folds by default')
    p.add_argument('--minutes', type=float, default=90, help='Per-fold GRPO window; expiry saves and pauses')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--micro-batch', type=int, default=2, help='Windows per rank; GLOBAL text batch remains 16')
    p.add_argument('--group-size', type=int, default=16)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--beta', type=float, default=.02)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--ce-weight', type=float, default=.1)
    p.add_argument('--out', type=Path, help='Separate run directory for a new fixed configuration')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(','); folds = [int(k) for k in a.folds.split(',')]
    if len(devices) != 4 or len(set(devices)) != 4 or not all(x.isdigit() for x in devices):
        p.error('Four distinct GPU IDs are required')
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
        p.error('Folds must be distinct members of 0,1,2,3,4')
    if (a.minutes <= 0 or min(a.epochs, a.micro_batch, a.updates) < 1 or a.group_size < 2
            or a.lr <= 0 or min(a.beta, a.ce_weight) < 0 or not 0 < a.epsilon < 1):
        p.error('Invalid GRPO configuration')
    runs = a.out.resolve() if a.out else ROOT / ('runs_grpo_smoke' if a.smoke else 'runs_grpo')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS='8')

    def execute(command):
        print('COMMAND', json.dumps(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not a.dry_run:
        from assets import originals
        originals()
        if not (BASE / 'vendor/native_q8.py').exists() or not (BASE / 'base_staged/tokenizer.json').exists():
            raise SystemExit('Prepare pinned public assets with python ../11/scripts/assets.py all')
    chosen = folds[:1] if a.smoke else folds
    for k in chosen:
        deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=a.minutes)
        execute([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 str(ROOT / 'scripts/grpo_train.py'), '--fold', str(k), '--out', str(runs),
                 '--deadline', deadline.isoformat(), '--epochs', str(a.epochs),
                 '--micro-batch', str(a.micro_batch), '--group-size', str(a.group_size),
                 '--updates', str(a.updates), '--lr', str(a.lr), '--beta', str(a.beta),
                 '--epsilon', str(a.epsilon), '--ce-weight', str(a.ce_weight)]
                + (['--max-batches', '1'] if a.smoke else []))
        validated = False
        if not a.dry_run and not a.smoke:
            from windows import sha
            evaluation = runs / f'fold{k}/native_eval.json'
            if evaluation.exists():
                record = json.loads(evaluation.read_text())
                validated = (record.get('parity_ok', False) and bool(record.get('artifact_sha256'))
                             and all(sha(evaluation.parent / n) == h for n, h in record['artifact_sha256'].items()))
        if not validated:
            execute([sys.executable, str(ROOT / 'scripts/evaluate.py'), '--run', str(runs / f'fold{k}'),
                     '--fold', str(k), '--gpus', '4'] + (['--smoke', '48'] if a.smoke else []))
    if not a.smoke:
        execute([sys.executable, str(ROOT / 'scripts/decide.py'), '--runs', str(runs),
                 '--folds', ','.join(map(str, chosen)), '--out', str(runs / 'decision.json')])


if __name__ == '__main__':
    main()
