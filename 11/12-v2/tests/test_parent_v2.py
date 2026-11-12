"""Verify actual pinned OOF voting math without loading a model or test data."""
import json
import runpy
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared import BASE
from public_parent import load_parent, WEIGHTS
from runtime_vote import multi_model_parent


def test():
    frame = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    classes = sorted(frame.model.unique())
    parent, record = load_parent(frame.prompt_id.values, classes)
    # common.public_lora_stack does not call native_q8. Supply only an empty
    # import placeholder so this CPU OOF check needs no downloaded native code.
    previous = sys.modules.get('native_q8')
    sys.modules['native_q8'] = types.ModuleType('native_q8')
    try:
        common = runpy.run_path(str(BASE / 'scripts/common.py'))
        geometric = common['public_lora_stack']()
    finally:
        if previous is None:
            sys.modules.pop('native_q8', None)
        else:
            sys.modules['native_q8'] = previous
    unique, inverse = np.unique(frame.prompt_id.values, return_inverse=True)
    shuffled = unique.copy(); np.random.RandomState(0).shuffle(shuffled)
    membership = {prompt: i % 5 for i, prompt in enumerate(shuffled)}
    split = np.array([membership[p] for p in unique])
    terms = {}
    for name, weight in WEIGHTS.items():
        with np.load(BASE / f'data/reference/{name}.npz', allow_pickle=False) as z:
            labels = z['labels'].astype(str).tolist()
            logits = z['lp'][:, :, [labels.index(c) for c in classes]].astype(np.float64)
        selected = np.asarray(record[name]['adapter_by_prompt_fold'])[split][inverse]
        terms[name] = weight * logits[selected, np.arange(len(frame))]
    runtime = multi_model_parent(np.exp(geometric), terms)
    assert np.allclose(parent, np.log(runtime), atol=5e-7)
    assert np.allclose(np.exp(parent).sum(-1), 1.)
    with tempfile.TemporaryDirectory(prefix='voter_hash_') as directory:
        root = Path(directory)
        for name in WEIGHTS:
            shutil.copy(BASE / f'data/reference/{name}.npz', root / f'{name}.npz')
        path = root / 'q14v5.npz'; path.write_bytes(path.read_bytes() + b'changed')
        try:
            load_parent(frame.prompt_id.values, classes, reference_dir=root)
        except ValueError as error:
            assert 'changed' in str(error)
        else:
            raise AssertionError('Changed voter OOF accepted')
    print(json.dumps(dict(actual_pinned_voters=3, rows=len(frame),
                          oof_runtime_vote_parity=True, voter_hash_guard=True)))


if __name__ == '__main__':
    test()
