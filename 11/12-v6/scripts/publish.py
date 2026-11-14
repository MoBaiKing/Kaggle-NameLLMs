"""Explicit Kaggle upload/commit/one-shot submission, with a persistent receipt.

No action runs at import. --submit is required to spend a competition submission;
ambiguous submission responses are recorded and never retried automatically.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, check=True, timeout=5400):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    for name in ('KAGGLE_API_TOKEN', 'KAGGLE_KEY'):
        secret = os.environ.get(name)
        if secret:
            output = output.replace(secret, '***')
    print(output[-2000:], flush=True)
    if check and result.returncode:
        raise RuntimeError(f'Kaggle command failed: {command[:3]}')
    return result.returncode, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=ROOT / 'submission_v6')
    p.add_argument('--submit', action='store_true', help='Commit and submit exactly once after checks')
    p.add_argument('--message', default='11/12 v6: native retrained on Open Pool v15 (public recipe, match_weight, fidelity) + seed soup; public fusion unchanged')
    a = p.parse_args()
    directory = a.directory.resolve()
    receipt = directory / 'receipt.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else {}

    def save(**values):
        state.update(values, updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(receipt)

    if state.get('status') in ('SUBMITTING', 'SUBMITTED', 'SUBMIT_UNCONFIRMED'):
        raise SystemExit('Submission already used or uncertain. Inspect the receipt and Kaggle history; do not resubmit blindly.')
    metadata = json.loads((directory / 'final/kernel-metadata.json').read_text())
    kernel = metadata['id']
    if not metadata['is_private'] or metadata['enable_internet']:
        raise ValueError('Expected a private Notebook with internet disabled')
    if not a.submit:
        print(json.dumps(dict(kernel=kernel, action='PREVIEW_ONLY', directory=str(directory),
                              next_command='python scripts/publish.py --submit')), flush=True)
        return
    upload = directory / 'upload'
    spec_path = upload / 'dyl_v6_native.json'
    if not spec_path.is_file():
        raise ValueError('Missing v6 package; run build_submission.py first')
    spec = json.loads(spec_path.read_text())
    if spec.get('method') != 'v6_native_data_soup' or sorted(spec.get('folds', {})) != [str(k) for k in range(5)]:
        raise ValueError('The upload must contain five v6 native fold states')
    if upload.exists():
        ds = json.loads((upload / 'dataset-metadata.json').read_text())['id']
        code, text = run(['kaggle', 'datasets', 'create', '-p', str(upload), '-r', 'skip'], check=False)
        if code and ('already exist' in text.lower() or 'conflict' in text.lower()):
            run(['kaggle', 'datasets', 'version', '-p', str(upload), '-r', 'skip', '-m', '11/12 v6 native fold states'])
        elif code:
            raise RuntimeError('Dataset creation failed; no competition submission sent')
        end = time.time() + 7200
        while True:
            _, text = run(['kaggle', 'datasets', 'status', ds], check=False, timeout=120)
            if 'ready' in text.lower():
                break
            if time.time() > end:
                raise TimeoutError('Dataset did not become ready')
            time.sleep(30)
    _, output = run(['kaggle', 'kernels', 'push', '-p', str(directory / 'final')], timeout=900)
    match = re.search(r'version (\d+) successfully pushed', output, re.I)
    if not match:
        raise RuntimeError('Notebook version was not confirmed; no submission sent')
    version = match.group(1); save(status='COMMITTING', kernel=kernel, kernel_version=version)
    end = time.time() + 9 * 3600
    while True:
        _, output = run(['kaggle', 'kernels', 'status', kernel], timeout=120)
        if 'COMPLETE' in output:
            break
        if any(word in output.upper() for word in ('ERROR', 'FAILED', 'CANCELLED')):
            raise RuntimeError('Notebook failed; no submission sent')
        if time.time() > end:
            raise TimeoutError('Notebook commit timed out; no submission sent')
        time.sleep(30)
    outputs = directory / 'commit_output'; outputs.mkdir(exist_ok=True)
    run(['kaggle', 'kernels', 'output', kernel, '-p', str(outputs)], timeout=1800)
    logs = '\n'.join(f.read_text(errors='replace') for f in outputs.rglob('*.log'))
    if 'DYL_V6_NATIVE_READY OK' not in logs:
        raise RuntimeError('Native slot setup was not confirmed; no submission sent')
    submission = outputs / 'submission.parquet'
    if not submission.is_file():
        raise FileNotFoundError('Notebook produced no submission.parquet')
    import pandas as pd
    frame = pd.read_parquet(submission)
    if list(frame.columns) != ['id', 'model'] or frame.isna().any().any() or not frame.id.is_unique:
        raise ValueError('Invalid submission schema')
    save(status='SUBMITTING')
    try:
        _, output = run(['kaggle', 'competitions', 'submit', 'name-that-llm',
                         '-k', kernel, '-v', version, '-f', 'submission.parquet', '-m', a.message],
                        check=False, timeout=900)
    except Exception:
        save(status='SUBMIT_UNCONFIRMED')
        raise
    match = re.search(r'Submission ref:\s*(\d+)', output)
    save(status='SUBMITTED' if match else 'SUBMIT_UNCONFIRMED',
         submission_ref=match.group(1) if match else None)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
