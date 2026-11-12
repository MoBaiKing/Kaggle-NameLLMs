"""Pinned, label-free OOF parent proxy; no native runtime or GPU is required."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from shared import BASE
from windows import sha

WEIGHTS = {'q14v5': 3., 'q8v5': 2., 'll8v5': 1.}


def lsm(value):
    value = value - value.max(-1, keepdims=True)
    return value - np.log(np.exp(value).sum(-1, keepdims=True))


def load_parent(prompts, classes, reference_dir=None, spec_path=None):
    """Match common.public_lora_stack using confidence, never true row labels.

Source files have no row IDs, so their full SHA256 and original train CSV hash
are mandatory. The trainer separately verifies that exact official row order.
"""
    reference_dir = Path(reference_dir) if reference_dir else BASE / 'data/reference'
    spec_path = Path(spec_path) if spec_path else BASE / 'config/assets.json'
    spec = json.loads(spec_path.read_text())
    prompts = np.asarray(prompts)
    unique, inverse = np.unique(prompts, return_inverse=True)
    shuffled = unique.copy(); np.random.RandomState(0).shuffle(shuffled)
    membership = {prompt: i % 5 for i, prompt in enumerate(shuffled)}
    split = np.array([membership[p] for p in unique])
    if any(not (split == k).any() for k in range(5)):
        raise ValueError('At least five prompt groups are required for the fixed parent fold mapping')
    counts = np.bincount(inverse, minlength=len(unique))
    score = np.zeros((len(prompts), len(classes)), np.float64)
    vote = np.zeros_like(score)
    records = {}
    for name, weight in WEIGHTS.items():
        path = reference_dir / f'{name}.npz'
        fingerprint = sha(path)
        if fingerprint != spec['reference_data'][path.name]['sha256']:
            raise ValueError(f'Public parent OOF changed: {path.name}')
        with np.load(path, allow_pickle=False) as z:
            labels = z['labels'].astype(str).tolist()
            if len(labels) != len(classes) or set(labels) != set(classes):
                raise ValueError('Public parent class order/set differs')
            logits = z['lp'].astype(np.float64)
        if logits.shape != (5, len(prompts), len(classes)) or not np.isfinite(logits).all():
            raise ValueError('Invalid public parent OOF shape or scores')
        logits = logits[:, :, [labels.index(c) for c in classes]]
        confidence = np.stack([np.bincount(inverse, logits[k].max(-1), len(unique)) / counts for k in range(5)])
        cost = np.array([[confidence[k, split == fold].mean() for k in range(5)] for fold in range(5)])
        row, column = linear_sum_assignment(cost)
        mapping = np.empty(5, dtype=int); mapping[row] = column
        selected = mapping[split][inverse]
        member = lsm(logits[selected, np.arange(len(prompts))])
        score += weight * member
        vote += weight * np.exp(member)
        records[name] = dict(sha256=fingerprint, adapter_by_prompt_fold=mapping.tolist(), weight=weight)
    geometric = np.exp(lsm(score / sum(WEIGHTS.values())))
    arithmetic = vote / sum(WEIGHTS.values())
    # The complete public ensemble replaces geometric at deployment. The OOF
    # files cover only these three members, so this is explicitly a proxy.
    parent = .75 * geometric + .25 * arithmetic
    return np.log(parent).astype(np.float32), records
