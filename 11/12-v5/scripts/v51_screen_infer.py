"""GPU screening inference from the actual pinned native/Qwen14B/Llama8B public weights.
For q14v5/ll8v5, --spec is the corresponding pinned Notebook MODELS entry with local asset paths.
Only its plain classifier is used; official labels/prompt means do not screen unverified passages.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v51_common import BASE, ROOT, CLASSES, sha, dump, save_npz, load_verified, lsm


def public_l24(namespace):
    from build_v51 import SRC, PIN
    notebook = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    if sha(notebook) != PIN:
        raise ValueError('Pinned screening runtime changed')
    cells = json.loads(notebook.read_text())['cells']
    source = next(''.join(c['source']).partition('\n')[2] for c in cells
                  if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n'))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'l24_predict')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned_l24.py', 'exec'), namespace)
    return namespace['l24_predict']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--member', choices=['native8b', 'q14v5', 'll8v5'], required=True)
    p.add_argument('--spec', type=Path, help='Local MODELS entry; required for q14v5/ll8v5')
    p.add_argument('--deadline', required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args(); end = dt.datetime.fromisoformat(a.deadline)
    if end.tzinfo is None:
        p.error('Timezone required')
    if time.time() >= end.timestamp() - 300:
        raise SystemExit(75)
    # Frozen screening has no optimizer to save; completed native chunks remain reusable.
    # The shared-server lease supervisor must additionally handle changed/cancelled reservations.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(75))
    signal.setitimer(signal.ITIMER_REAL, max(1., end.timestamp() - time.time() - 180))
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Screening inference requires an authorized GPU, no CPU fallback')
    sys.path.insert(0, str(BASE / 'vendor'))
    source = ROOT / 'cache/open_pool_latest/gemini37_unverified.parquet'
    expected = json.loads((ROOT / 'artifacts/v51_plan.json').read_text())['pool_files'][source.name]
    if sha(source) != expected:
        raise ValueError('Unverified source snapshot changed')
    frame = pd.read_parquet(source); out = a.out or ROOT / 'cache' / f'{a.member}_screen.npz'
    if out.with_suffix('.json').exists():
        record = load_verified(out)
        if record['source_sha256'] != expected or record['model_id'] != a.member:
            raise ValueError('Cached screening identity changed')
        print('Verified screening reused', out); return
    if a.member == 'native8b':
        from v51_collect import verify_assets, NATIVE
        import native_q8
        assets = verify_assets()
        model = native_q8.NativeQ8(BASE / 'base_staged', NATIVE, ROOT / 'cache/screen_native_work', gpus=1)
        selections = [json.loads((NATIVE / f'calibration/fold{k}.json').read_text())['models']['qwen8'] for k in range(5)]
        result = []; chunks = ROOT / 'cache/screen_native_chunks'; chunks.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(frame), 256):
            path = chunks / f'{start}.npz'
            if path.with_suffix('.json').exists():
                load_verified(path)
                with np.load(path, allow_pickle=False) as z:
                    result.append(z['probs'])
                continue
            if time.time() >= end.timestamp() - 300:
                raise SystemExit(75)
            res, finite = model.predict(frame.text.iloc[start:start + 256].tolist(), folds=range(5), batch_size=4)
            if not finite.all() or res['folds'].tolist() != list(range(5)):
                raise ValueError('Incomplete native screening')
            probs = np.mean([np.exp(lsm(res['plain'][k] * selections[k]['plain_scale'])) for k in range(5)], axis=0)
            save_npz(path, probs=probs); dump(path.with_suffix('.json'), dict(payload_sha256=sha(path), source_sha256=expected))
            result.append(probs)
        probs = np.concatenate(result); identity = assets
    else:
        if not a.spec:
            p.error('--spec required: copy the pinned MODELS entry and resolve its base/adapter paths locally')
        spec = json.loads(a.spec.read_text())
        expected_keep = 27 if a.member == 'q14v5' else 24
        if spec['name'] != a.member or spec['keep'] != expected_keep or spec['layers'] != [2, 4, 6, 8, 12, 16, 20, 24] or spec['max_len'] != 384:
            raise ValueError('Screening model differs from pinned public architecture')
        # Hash model and head/adapter inputs, not just the path or an asserted human-readable name.
        files = [f for directory in [spec['path'], *spec['adapters']] for f in Path(directory).rglob('*')
                 if f.is_file() and f.suffix in ('.json', '.safetensors', '.pt', '.txt', '.model')]
        identity = {str(f.resolve()): sha(f) for f in sorted(set(files))}
        if not files:
            raise ValueError('No real public model files found')
        import gc
        from transformers import AutoModel, AutoTokenizer
        from pipe import pipelined, spans_devices
        namespace = dict(torch=torch, np=np, os=os, json=json, gc=gc, LABELS=CLASSES,
                         AutoModel=AutoModel, AutoTokenizer=AutoTokenizer, pipelined=pipelined,
                         spans_devices=spans_devices, log=print)
        predict = public_l24(namespace)
        lp, _ = predict(spec, frame.text.fillna('').values)
        probs = np.exp(lsm(lp))
    if probs.shape != (len(frame), 18) or not np.isfinite(probs).all():
        raise ValueError('Invalid screening result')
    checkpoint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    save_npz(out, ids=frame.id.astype(str).values, labels=np.array(CLASSES), probs=probs)
    dump(out.with_suffix('.json'), dict(payload_sha256=sha(out), source_sha256=expected,
         model_id=a.member, checkpoint_sha256=checkpoint, files=identity, scope='unverified_external_GPU_inference'))


if __name__ == '__main__':
    main()
