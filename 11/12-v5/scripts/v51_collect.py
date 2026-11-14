"""Fourth GPU: frozen topic embeddings and five native fold outputs, resumable in bounded chunks."""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v51_common import BASE, ROOT, RUN, dump, sha, save_npz, load_verified

NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'


def verify_assets():
    spec = json.loads((BASE / 'config/assets.json').read_text())
    if sha(BASE / 'data/original/train.csv') != spec['original_data']['train.csv']['sha256'] or sha(BASE / 'data/folds.json') != spec['folds_sha256']:
        raise ValueError('Official data/folds changed')
    if sha(NATIVE / 'manifest.json') != spec['native_assets']['manifest_sha256']:
        raise ValueError('Native manifest changed')
    manifest = json.loads((NATIVE / 'manifest.json').read_text())
    checked = {}
    for k in range(5):
        for rel in (f'fold{k}/trainables.safetensors', f'calibration/fold{k}.json'):
            checked[rel] = sha(NATIVE / rel)
            if checked[rel] != manifest['file_hashes'][rel]:
                raise ValueError(f'Native asset changed: {rel}')
    for name in ('native_q8.py', 'qwen_groups.py'):
        checked[name] = sha(BASE / 'vendor' / name)
    if checked['native_q8.py'] != spec['public_notebook']['native_code_sha256']:
        raise ValueError('Native runtime changed')
    notebook = BASE / 'external/pinned_notebook/ntl-684-ozertuu-mirror.ipynb'
    if sha(notebook) != spec['public_notebook']['sha256']:
        raise ValueError('Pinned Notebook changed')
    cells = json.loads(notebook.read_text())['cells']
    code = next(''.join(c['source']).partition('\n')[2] for c in cells
                if ''.join(c.get('source', [])).startswith('%%writefile qwen_groups.py\n'))
    if hashlib.sha256(code.encode()).hexdigest() != checked['qwen_groups.py']:
        raise ValueError('Topic embedder differs from the deployed public code')
    index = json.loads((BASE / 'base_staged/model.safetensors.index.json').read_text())
    # Record exact local shards; snapshot_download pins the HF revision before this verifier.
    base = {n: sha(BASE / 'base_staged' / n) for n in sorted(set(index['weight_map'].values()))}
    for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
        base[n] = sha(BASE / 'base_staged' / n)
        if base[n] != spec['native_assets']['files']['base/' + n]:
            raise ValueError('Staged base/tokenizer differs from pinned native base')
    return dict(base=base, native=checked, revision=spec['model']['revision'])


def eb_rows(ext, minimum_prompts=1000):
    """Whole, diverse panels, deterministic 60k target. A panel is never truncated to fit the budget."""
    count = ext.groupby('prompt').model.nunique()
    groups = count[count >= 4].index.values
    groups = np.random.default_rng(42).permutation(groups)
    chosen, total = [], 0
    sizes = ext.groupby('prompt').size()
    for group in groups:
        if total >= 60000 and len(chosen) >= minimum_prompts:
            break
        chosen.append(group); total += int(sizes[group])
    if len(chosen) < minimum_prompts:
        raise ValueError('Need at least 1000 external topic panels; do not claim adequate topic coverage')
    return ext[ext.prompt.isin(chosen)].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external', type=Path, default=ROOT / 'cache/external_v51.parquet')
    p.add_argument('--out', type=Path, default=RUN / 'collect')
    p.add_argument('--deadline', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); end = dt.datetime.fromisoformat(a.deadline)
    if end.tzinfo is None:
        p.error('Timezone required')
    deadline = end.timestamp()
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Collector must see exactly one authorized GPU')
    assets = verify_assets(); ext_rec = load_verified(a.external)
    ext = pd.read_parquet(a.external)
    # Use all classes in topic panels, independent of experts' class subsets.
    ext = eb_rows(ext) if not a.smoke else ext.iloc[:32]
    off = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    frame = pd.concat([off[['id', 'text']], ext[['id', 'text']]], ignore_index=True)
    if a.smoke:
        frame = frame.iloc[:32]
    identity = dict(assets=assets, external=ext_rec['payload_sha256'], rows=len(frame), smoke=a.smoke,
                    code_sha256=sha(__file__))
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'identity.json').exists() and json.loads((a.out / 'identity.json').read_text()) != identity:
        raise ValueError('Collector inputs changed; use a new run directory')
    dump(a.out / 'identity.json', identity)
    query = a.out / 'query.csv'; frame.to_csv(query, index=False)
    # Run exact public qwen_groups code, including its tokenizer, truncation, eos and fp16 output.
    if not (a.out / 'topic.json').exists():
        if time.time() >= deadline - 300:
            raise SystemExit(75)
        subprocess.run([sys.executable, str(BASE / 'vendor/qwen_groups.py'), '--base', str(BASE / 'base_staged'),
                        '--assets', str(NATIVE), '--query', str(query), '--out', str(a.out / 'topic.npy'),
                        '--work', str(a.out / 'topic_work')], check=True)
        dump(a.out / 'topic.json', dict(sha256=sha(a.out / 'topic.npy'), query_sha256=sha(query), identity=identity))
    topic_rec = json.loads((a.out / 'topic.json').read_text())
    if sha(a.out / 'topic.npy') != topic_rec['sha256'] or sha(query) != topic_rec['query_sha256']:
        raise ValueError('Topic cache changed')
    sys.path.insert(0, str(BASE / 'vendor'))
    import native_q8
    model = native_q8.NativeQ8(BASE / 'base_staged', NATIVE, a.out / 'native_work', gpus=1)
    arrays = []
    for k in range(5):
        pieces = []
        for start in range(0, len(frame), 256):
            out = a.out / f'fold{k}_chunk{start}.npz'
            if out.with_suffix('.json').exists():
                load_verified(out)
            else:
                if time.time() >= deadline - 300:
                    raise SystemExit(75)
                res, finite = model.predict(frame.text.iloc[start:start + 256].tolist(), folds=[k], batch_size=4)
                if not finite.all() or res['folds'].tolist() != [k]:
                    raise ValueError('Incomplete/non-finite native chunk')
                save_npz(out, plain=res['plain'][0], centered=res['centered'][0], bias=res['centered_bias'][0])
                dump(out.with_suffix('.json'), dict(payload_sha256=sha(out), identity=identity))
            with np.load(out, allow_pickle=False) as z:
                pieces.append((z['plain'], z['centered'], z['bias']))
        arrays.append(dict(plain=np.concatenate([p[0] for p in pieces]), centered=np.concatenate([p[1] for p in pieces]),
                           bias=pieces[0][2]))
    topic = np.load(a.out / 'topic.npy')
    if topic.shape[0] != len(frame) or not np.isfinite(topic).all():
        raise ValueError('Invalid topic vectors')
    out = a.out / 'features.npz'
    save_npz(out, topic=topic, plain=np.stack([x['plain'] for x in arrays]),
             centered=np.stack([x['centered'] for x in arrays]), bias=np.stack([x['bias'] for x in arrays]),
             ids=frame.id.astype(str).values, external_groups=ext.prompt.astype(str).values,
             external_y=ext.label.values, official_rows=np.array(len(off)))
    dump(out.with_suffix('.json'), dict(payload_sha256=sha(out), identity=identity, smoke=a.smoke,
                                      official_sha256=sha(BASE / 'data/original/train.csv')))


if __name__ == '__main__':
    main()
