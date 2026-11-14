"""Four independent GPU jobs: three sibling experts plus EB frozen inference.
A lease command may change GPU IDs/end time on every poll. Revocation saves and releases jobs.
The command must emit JSON {allowed: bool, gpus: [0,1,2,3], deadline: timezone-aware ISO}.
"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from v51_common import ROOT, RUN, GROUPS, dump, sha, load_verified


def parse_lease(data):
    end = dt.datetime.fromisoformat(data['deadline'])
    if end.tzinfo is None:
        raise ValueError('Lease deadline must contain timezone')
    gpus = [str(g) for g in data['gpus']]
    if len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        raise ValueError('Invalid lease GPU IDs')
    return dict(allowed=bool(data['allowed']) and time.time() < end.timestamp() - 180,
                gpus=gpus, deadline=end.isoformat(), end=end.timestamp())


def jobs(smoke, root, deadline):
    extra = ['--smoke'] if smoke else []
    return [(g, [sys.executable, str(ROOT / 'scripts/v51_train.py'), '--group', g, '--out', str(root / 'experts'),
                 '--deadline', deadline, *extra]) for g in GROUPS] + [
           ('eb', [sys.executable, str(ROOT / 'scripts/v51_collect.py'), '--out', str(root / 'collect'),
                   '--deadline', deadline, *extra])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--deadline', help='Teammate dedicated allocation end, timezone-aware')
    p.add_argument('--lease-command', type=Path, help='JSON argv file; queries current reservation AND idle authorized GPUs')
    p.add_argument('--out', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-evaluate', action='store_true')
    a = p.parse_args(); root = a.out or (ROOT / 'runs_v51_smoke' if a.smoke else RUN)
    if not a.deadline and not a.lease_command and not a.dry_run:
        p.error('Provide a live lease command or the explicitly authorized dedicated allocation deadline')

    def lease():
        if a.lease_command:
            command = json.loads(a.lease_command.read_text())
            if not isinstance(command, list) or not all(isinstance(c, str) for c in command):
                raise ValueError('Lease command must be a JSON argument list')
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
            return parse_lease(json.loads(result.stdout))
        return parse_lease(dict(allowed=True, gpus=a.gpus.split(','),
                          deadline=a.deadline or '2099-01-01T00:00:00+08:00'))

    initial = lease()
    if a.dry_run:
        print(json.dumps(dict(root=str(root), jobs=jobs(a.smoke, root, initial['deadline']),
                         scheduling='three one-GPU experts + one one-GPU collector; resumable'), indent=2)); return
    load_verified(ROOT / 'cache/external_v51.parquet')
    root.mkdir(parents=True, exist_ok=True)
    # Lock excludes duplicate launchers. Never acquire GPU memory just to wait for a lease.
    import fcntl
    lock = (root / 'launcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = [g for g, _ in jobs(a.smoke, root, initial['deadline'])]; active = {}; complete = []
    pending.sort(key=lambda n: not ((root / 'collect' if n == 'eb' else root / 'experts' / n).exists()))
    stopping = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    # Reuse only verified completion. Completed tasks have priority over starting duplicate work.
    def finished(name):
        if name == 'eb':
            path = root / 'collect/features.npz'
            if path.with_suffix('.json').exists():
                load_verified(path); return True
        else:
            from v51_train import excluded_sets, split_name
            for excluded in excluded_sets(a.smoke):
                path = root / 'experts' / name / split_name(excluded) / 'scores.npz'
                if not path.with_suffix('.json').exists():
                    return False
                rec = load_verified(path)
                completion = path.parent / 'complete.json'
                if not completion.exists():
                    return False
                if json.loads(completion.read_text()) != rec:
                    raise ValueError('Expert completion record differs from scored checkpoint')
                if sha(path.parent / 'final.safetensors') != rec['checkpoint_sha256']:
                    raise ValueError('Completed expert checkpoint changed')
            return True
        return False

    def terminate(name, hard=False):
        proc = active[name]['proc']
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
            active[name].setdefault('stopping_at', time.time())

    try:
        while pending or active:
            try:
                current = lease()
            except Exception as exc:
                current = dict(allowed=False, gpus=[], end=0, deadline=initial['deadline'])
                print('Lease query failed; pausing:', repr(exc), flush=True)
            for name in list(active):
                item = active[name]; proc = item['proc']
                revoked = not current['allowed'] or item['gpu'] not in current['gpus'] or time.time() > current['end'] - 180
                if revoked or stopping['flag']:
                    terminate(name)
                if item.get('stopping_at') and time.time() - item['stopping_at'] > 90:
                    terminate(name, hard=True)
                code = proc.poll()
                if code is not None:
                    item['log'].close(); del active[name]
                    if code == 0 and finished(name):
                        complete.append(name)
                    elif code in (75, -signal.SIGTERM, -signal.SIGKILL):
                        pending.insert(0, name)
                    else:
                        dump(root / 'status.json', dict(status='FAILED', job=name, exit_code=code, completed=complete))
                        raise RuntimeError(f'{name} failed ({code}); see its log')
            if stopping['flag'] and not active:
                dump(root / 'status.json', dict(status='PAUSED', pending=pending, completed=complete)); return
            if current['allowed'] and not stopping['flag']:
                busy = {item['gpu'] for item in active.values()}
                available = [g for g in current['gpus'] if g not in busy]
                while pending and available:
                    name = pending.pop(0)
                    if finished(name):
                        complete.append(name); continue
                    gpu = available.pop(0)
                    command = dict(jobs(a.smoke, root, current['deadline']))[name]
                    log = (root / f'{name}.log').open('a', buffering=1)
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false'))
                    active[name] = dict(proc=proc, gpu=gpu, log=log)
            dump(root / 'status.json', dict(status='RUNNING' if active else 'PAUSED', completed=complete,
                 pending=pending, active={n: dict(pid=i['proc'].pid, gpu=i['gpu']) for n, i in active.items()},
                 lease=current, updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))
            if pending or active:
                time.sleep(10)
    finally:
        for name in list(active):
            terminate(name)
        for name in list(active):
            try:
                active[name]['proc'].wait(timeout=90)
            except subprocess.TimeoutExpired:
                terminate(name, hard=True); active[name]['proc'].wait()
            active[name]['log'].close()
    if not a.smoke and not a.skip_evaluate:
        subprocess.run([sys.executable, str(ROOT / 'scripts/v51_evaluate.py'), '--collect', str(root / 'collect/features.npz'),
                        '--experts', str(root / 'experts'), '--out', str(root / 'evidence')], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'))
    dump(root / 'status.json', dict(status='SMOKE_DONE' if a.smoke else 'VALIDATED', completed=complete,
                                   hidden_test_evaluated=False))


if __name__ == '__main__':
    main()
