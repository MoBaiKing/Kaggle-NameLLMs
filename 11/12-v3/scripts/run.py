"""Four cooperating GPUs, folds in sequence: OOF targets (CPU) -> v3 continuation -> native GPU validation;
optional v2 comparison (diagnostics only); then the proxy evidence report (decide_v3.py). No public fallback.
Re-running the same command resumes: finished folds and passed evaluations are verified and reused."""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from shared import BASE, ROOT
import oof_targets as T

PAUSED = 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--folds', default='0,1,2,3,4', help='All five are needed for the gate')
    p.add_argument('--minutes', type=float, default=90, help='Per-fold training window; expiry saves and pauses')
    p.add_argument('--micro-batch', type=int, default=2, help='Windows per rank; GLOBAL text batch remains 16')
    p.add_argument('--v2-runs', type=Path, default=None, help='Finished 11/12-v2/runs_grpo, evaluated for comparison only (costs 5 extra GPU evaluations)')
    p.add_argument('--out', type=Path, help='Separate run directory for a new fixed configuration')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(','); folds = [int(k) for k in a.folds.split(',')]
    if len(devices) != 4 or len(set(devices)) != 4 or not all(x.isdigit() for x in devices):
        p.error('Four distinct GPU IDs are required')
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
        p.error('Folds must be distinct members of 0,1,2,3,4')
    runs = a.out.resolve() if a.out else ROOT / ('runs_v3_smoke' if a.smoke else 'runs_v3')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS='8')

    def execute(command, cpu=False):
        print('COMMAND', json.dumps(command), flush=True)
        if a.dry_run:
            return 0
        code = subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES='') if cpu else env).returncode
        if code == PAUSED:
            print(json.dumps(dict(status='PAUSED', note='window ended; checkpoint saved; rerun the same command to resume')), flush=True)
            raise SystemExit(PAUSED)
        if code:
            raise SystemExit(f'Step failed with exit code {code}: {command[1] if len(command) > 1 else command}')
        return code

    if not a.dry_run:
        from assets import originals
        originals()
        if not (BASE / 'vendor/native_q8.py').exists() or not (BASE / 'base_staged/tokenizer.json').exists():
            raise SystemExit('Prepare pinned public assets with python ../11/scripts/assets.py all')
    targets = T.DEFAULT
    if not targets.exists():
        execute([sys.executable, str(ROOT / 'scripts/oof_targets.py'), '--out', str(targets)], cpu=True)
    else:
        execute([sys.executable, str(ROOT / 'scripts/oof_targets.py'), '--out', str(targets), '--check'], cpu=True)
    chosen = folds[:1] if a.smoke else folds
    for k in chosen:
        deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=a.minutes)
        execute([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 str(ROOT / 'scripts/train_v3.py'), '--fold', str(k), '--out', str(runs), '--targets', str(targets),
                 '--deadline', deadline.isoformat(), '--micro-batch', str(a.micro_batch)]
                + (['--max-batches', '2'] if a.smoke else []))
        evaluated = runs / f'fold{k}/native_eval.json'
        if a.smoke or a.dry_run or not evaluated.exists():
            execute([sys.executable, str(ROOT / 'scripts/evaluate_v3.py'), '--run', str(runs / f'fold{k}'),
                     '--fold', str(k), '--gpus', '4'] + (['--smoke', '48'] if a.smoke else []))
    if a.smoke:
        print(json.dumps(dict(status='SMOKE_DONE', runs=str(runs))), flush=True)
        return
    diagnostics = None
    if a.v2_runs:
        diagnostics = runs / 'v2_diagnostics'
        execute([sys.executable, str(ROOT / 'scripts/wise.py'), '--from-v2', str(a.v2_runs.resolve()),
                 '--out', str(diagnostics)], cpu=True)
        for k in range(5):
            if a.dry_run or not (diagnostics / f'fold{k}/native_eval.json').exists():
                execute([sys.executable, str(ROOT / 'scripts/evaluate_v3.py'), '--run', str(diagnostics / f'fold{k}'),
                         '--fold', str(k), '--gpus', '4'])
    execute([sys.executable, str(ROOT / 'scripts/decide_v3.py'), '--runs', str(runs), '--targets', str(targets),
             '--out', str(runs / 'decision.json')] + (['--v2-diagnostics', str(diagnostics)] if diagnostics else []), cpu=True)


if __name__ == '__main__':
    main()
