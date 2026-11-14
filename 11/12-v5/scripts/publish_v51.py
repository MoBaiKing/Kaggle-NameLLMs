"""Explicit one-shot Kaggle submission after accepted-component build. GitHub push does not invoke this."""
import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path

from v51_common import ROOT, dump, sha
from build_v51 import validate_decision, SPEC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=ROOT / 'submission_v51')
    p.add_argument('--submit', action='store_true')
    a = p.parse_args(); directory = a.directory.resolve()
    build = json.loads((directory / 'build.json').read_text())
    validate_decision(build['decision'])
    if sha(build['decision']) != build['decision_sha256'] or sha(directory / 'final/submission.ipynb') != build['notebook_sha256']:
        raise ValueError('Notebook or decision changed after build')
    for name, expected in build['sha256'].items():
        if sha(directory / 'upload' / name) != expected:
            raise ValueError('Deployment bundle changed')
    metadata = json.loads((directory / 'final/kernel-metadata.json').read_text())
    if not metadata['is_private'] or metadata['enable_internet'] or metadata['id'] != build['kernel']:
        raise ValueError('Unexpected kernel metadata')
    spec = json.loads((directory / 'upload' / SPEC).read_text())
    if spec['enabled'] != build['enabled'] or spec['sha256'] != build['sha256']:
        raise ValueError('Inconsistent bundle gate')
    if not a.submit:
        print(json.dumps(dict(action='PREVIEW_ONLY', **build), indent=2)); return
    receipt = directory / 'receipt.json'
    if receipt.exists():
        raise ValueError('A publication receipt already exists; inspect Kaggle history before any retry')
    # Failures/ambiguous responses remain recorded. Never silently consume a second submission.
    dump(receipt, dict(status='PUBLISHING', kernel=build['kernel'], decision_sha256=build['decision_sha256']))
    import shared  # noqa: F401
    from publish import run  # v4 CLI helper: timeout, captured output, secret redaction
    run(['kaggle', 'datasets', 'create', '-p', str(directory / 'upload'), '-r', 'skip'])
    end = time.time() + 7200
    while True:
        _, text = run(['kaggle', 'datasets', 'status', build['dataset']], check=False, timeout=120)
        if 'ready' in text.lower():
            break
        if time.time() > end:
            raise TimeoutError('Dataset readiness timed out')
        time.sleep(30)
    _, text = run(['kaggle', 'kernels', 'push', '-p', str(directory / 'final')], timeout=900)
    match = re.search(r'version (\d+) successfully pushed', text, re.I)
    if not match:
        raise ValueError('Kernel version unconfirmed; inspect receipt/history')
    version = match.group(1)
    dump(receipt, dict(status='COMMITTING', kernel=build['kernel'], version=version))
    end = time.time() + 9 * 3600
    while True:
        _, text = run(['kaggle', 'kernels', 'status', build['kernel']], timeout=120)
        if 'COMPLETE' in text.upper():
            break
        if any(w in text.upper() for w in ('ERROR', 'FAILED', 'CANCELLED')):
            raise RuntimeError('Kaggle inference failed; no competition submission sent')
        if time.time() > end:
            raise TimeoutError('Kaggle inference timed out')
        time.sleep(30)
    out = directory / 'commit_output'; out.mkdir(exist_ok=True)
    run(['kaggle', 'kernels', 'output', build['kernel'], '-p', str(out)], timeout=1800)
    logs = '\n'.join(p.read_text(errors='replace') for p in out.rglob('*.log'))
    if 'DYL_V51_READY' not in logs or not (out / 'submission.parquet').exists():
        raise ValueError('Accepted v5.1 completion/output not confirmed')
    import pandas as pd
    df = pd.read_parquet(out / 'submission.parquet')
    if list(df.columns) != ['id', 'model'] or df.isna().any().any() or not df.id.is_unique:
        raise ValueError('Invalid submission schema')
    dump(receipt, dict(status='SUBMITTING', kernel=build['kernel'], version=version))
    try:
        _, text = run(['kaggle', 'competitions', 'submit', 'name-that-llm', '-k', build['kernel'], '-v', version,
                       '-f', 'submission.parquet', '-m', '11/12 v5.1: accepted EB / 8B sibling experts'], check=False, timeout=900)
        match = re.search(r'Submission ref:\s*(\d+)', text)
        dump(receipt, dict(status='SUBMITTED' if match else 'SUBMIT_UNCONFIRMED', kernel=build['kernel'], version=version,
             submission_ref=match.group(1) if match else None, updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    except Exception:
        dump(receipt, dict(status='SUBMIT_UNCONFIRMED', kernel=build['kernel'], version=version)); raise


if __name__ == '__main__':
    main()
