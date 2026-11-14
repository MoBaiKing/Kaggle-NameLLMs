"""Small, CPU-only protocol utilities; no model or dataset is loaded at import time."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / '11'
RUN = ROOT / 'runs_v51'
CLASSES = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
           'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
           'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
           'kimi-k3', 'qwen3.8-max']
GROUPS = {'gemini': CLASSES[6:9], 'glm_claude': [CLASSES[0], *CLASSES[9:11]],
          'gpt_kimi': [*CLASSES[11:15], CLASSES[16]]}
FAMILY = np.unique([c.split('-')[0] for c in CLASSES], return_inverse=True)[1]
SEED = 42


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); os.replace(temp, path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.npz')
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in arrays.items():
        if value.dtype.hasobject:
            if not all(isinstance(x, (str, np.str_)) for x in value.flat):
                raise ValueError(f'Object payload cannot be stored without pickle: {name}')
            arrays[name] = value.astype(str)
    np.savez(temp, **arrays); os.replace(temp, path)


def lsm(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def row_folds(frame, split):
    fold = np.full(len(frame), -1, int)
    for k, rows in split.items():
        rows = np.asarray(rows, int)
        if (fold[rows] >= 0).any():
            raise ValueError('Overlapping validation rows')
        fold[rows] = int(k)
    if set(fold) != set(range(5)):
        raise ValueError('Require complete five-fold assignment')
    for p in frame.prompt_id.unique():
        if len(np.unique(fold[frame.prompt_id.values == p])) != 1:
            raise ValueError('A prompt crosses folds')
    return fold


def sparse_groups(prompt, seed, maximum=2):
    rng = np.random.default_rng(seed); groups = np.empty(len(prompt), int); next_id = 0
    for p in np.unique(prompt):
        rows = rng.permutation(np.flatnonzero(prompt == p)); pos = 0
        while pos < len(rows):
            size = int(rng.integers(1, maximum + 1))
            groups[rows[pos:pos + size]] = next_id; next_id += 1; pos += size
    return groups


def ba(y, correct, weights=None):
    weights = np.ones(len(y)) if weights is None else weights
    return float(np.mean([np.average(correct[y == c], weights=weights[y == c])
                          for c in np.unique(y) if weights[y == c].sum()]))


def conditional(y, pred, weights=None):
    weights = np.ones(len(y)) if weights is None else weights
    family_ok = FAMILY[y] == FAMILY[pred]
    den = weights[family_ok].sum()
    return float(weights[pred == y].sum() / den) if den else 0.


def paired_ci(y, prompts, base, new, metric='ba', draws=4000, seed=42):
    """Paired cluster bootstrap. BA accepts mean per-row correctness across grouping draws.
    Conditional accuracy takes integer decisions and resamples its numerator AND denominator.
    Units are percentage points; these are official-train validation, never hidden-test scores.
    """
    y, prompts = np.asarray(y), np.asarray(prompts)
    _, inverse = np.unique(prompts, return_inverse=True); n = inverse.max() + 1
    score = (lambda p, w: ba(y, p, w)) if metric == 'ba' else (lambda p, w: conditional(y, p, w))
    baseline, candidate = score(base, None), score(new, None)
    rng = np.random.default_rng(seed); differences = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
        if any(weights[y == c].sum() == 0 for c in np.unique(y)):
            continue
        differences.append(100 * (score(new, weights) - score(base, weights)))
    if len(differences) < draws * .9:
        raise ValueError('Too few valid prompt bootstrap draws')
    return dict(base=100 * baseline, candidate=100 * candidate, delta=100 * (candidate - baseline),
                ci95=np.quantile(differences, [.025, .975]).tolist(), bootstrap_draws=len(differences),
                unit='percentage_points', metric=metric)


def load_verified(path):
    path = Path(path); record = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != record['payload_sha256']:
        raise ValueError(f'Payload changed: {path}')
    return record
