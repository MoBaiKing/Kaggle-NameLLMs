"""Exact native windows, deterministic token-weighted sampling and shared mmap tokens."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from shared import BASE, ROOT

INSTRUCTION = 'Instruct: Identify the language model that generated the text based on its writing style.\nQuery: '
LIMIT, TERMINAL = 512, 151643


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(data, stream, model):
    model = Path(model)
    files = ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'merges.txt', 'vocab.json']
    return dict(format=1, train=sha(Path(data) / 'train.csv'), stream=sha(stream),
                tokenizer={name: sha(model / name) for name in files},
                transformers=importlib.metadata.version('transformers'),
                tokenizers=importlib.metadata.version('tokenizers'),
                instruction=INSTRUCTION, limit=LIMIT, terminal=TERMINAL)


def write_cache(texts, tokenizer, target, fingerprint, batch=128):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / 'manifest.json').exists():
            store = TokenStore(target)
            if store.manifest['identity'] != fingerprint or len(store) != len(texts):
                raise ValueError('Token cache inputs changed; use another cache directory')
            return store
        if target.exists():
            raise ValueError('Incomplete token cache; inspect it before rebuilding')
        prefix = tokenizer(INSTRUCTION, add_special_tokens=False)['input_ids']
        if tokenizer('', add_special_tokens=True)['input_ids'] != [TERMINAL] or not 0 < len(prefix) < LIMIT - 1:
            raise ValueError('Unexpected native tokenizer')
        temp = Path(tempfile.mkdtemp(prefix=target.name + '.', dir=target.parent))
        try:
            offsets, count = [0], 0
            with (temp / 'tokens.bin').open('wb') as out:
                for start in range(0, len(texts), batch):
                    full = tokenizer([INSTRUCTION + str(t) for t in texts[start:start + batch]],
                                     truncation=False, add_special_tokens=False)['input_ids']
                    for tokens in full:
                        if not tokens:
                            raise ValueError('Empty encoded input')
                        np.asarray(tokens, dtype='<i4').tofile(out)
                        count += len(tokens); offsets.append(count)
            np.save(temp / 'offsets.npy', np.asarray(offsets, dtype=np.int64))
            manifest = dict(identity=fingerprint, rows=len(texts), prefix=prefix,
                            tokens=count, token_sha256=sha(temp / 'tokens.bin'),
                            offsets_sha256=sha(temp / 'offsets.npy'))
            (temp / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
            os.replace(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return TokenStore(target)


class TokenStore:
    def __init__(self, directory):
        path = Path(directory)
        self.manifest = json.loads((path / 'manifest.json').read_text())
        for filename, field in [('tokens.bin', 'token_sha256'), ('offsets.npy', 'offsets_sha256')]:
            if sha(path / filename) != self.manifest[field]:
                raise ValueError(f'Changed token cache: {filename}')
        self.tokens = np.memmap(path / 'tokens.bin', mode='r', dtype='<i4')
        self.offsets = np.load(path / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.prefix = list(self.manifest['prefix'])
        assert len(self.offsets) == self.manifest['rows'] + 1
        assert self.offsets[0] == 0 and self.offsets[-1] == len(self.tokens)
        assert (np.diff(self.offsets) > 0).all()

    def __len__(self):
        return self.manifest['rows']

    def full(self, row):
        return self.tokens[self.offsets[row]:self.offsets[row + 1]]

    def all_windows(self, row):
        tokens = self.full(row)
        capacity = LIMIT - len(self.prefix) - 1
        windows = [tokens[:LIMIT - 1].tolist() + [TERMINAL]]
        weights = [max(1, min(len(tokens), LIMIT - 1) - len(self.prefix))]
        for pos in range(LIMIT - 1, len(tokens), capacity):
            chunk = tokens[pos:pos + capacity].tolist()
            windows.append(self.prefix + chunk + [TERMINAL]); weights.append(len(chunk))
        return windows, np.asarray(weights, dtype=np.float64)

    def select(self, row, seed=42, step=0, stage=1, first=False):
        tokens = self.full(row)
        index = 0
        capacity = LIMIT - len(self.prefix) - 1
        if not first and len(tokens) > LIMIT - 1:
            first_weight = max(1, LIMIT - 1 - len(self.prefix))
            total = first_weight + len(tokens) - (LIMIT - 1)
            key = f'{seed}:{stage}:{step}:{int(row)}'.encode()
            # Integer sampling avoids floating-point boundary differences.
            draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little') % total
            if draw >= first_weight:
                index = 1 + (draw - first_weight) // capacity
        if index == 0:
            return tokens[:LIMIT - 1].tolist() + [TERMINAL]
        pos = LIMIT - 1 + (index - 1) * capacity
        return self.prefix + tokens[pos:pos + capacity].tolist() + [TERMINAL]


def prepare(data, stream, model, target):
    fingerprint = identity(data, stream, model)
    if (Path(target) / 'manifest.json').exists():
        store = TokenStore(target)
        if store.manifest['identity'] != fingerprint:
            raise ValueError('Token cache fingerprint differs from current inputs')
        return store
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    train = pd.read_csv(Path(data) / 'train.csv', dtype=str)
    pool = pd.read_parquet(stream, columns=['text'])
    texts = np.concatenate([train.text.values, pool.text.values])
    return write_cache(texts, tokenizer, target, fingerprint)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=BASE / 'data/original')
    p.add_argument('--stream', type=Path, default=BASE / 'data/stream_8b_cap9500.parquet')
    p.add_argument('--model', type=Path, default=BASE / 'base_staged')
    p.add_argument('--out', type=Path, default=ROOT / 'cache/tokens')
    a = p.parse_args()
    result = prepare(a.data, a.stream, a.model, a.out)
    print(json.dumps(dict(rows=len(result), tokens=result.manifest['tokens'], ready=True)))
