"""The proxy's fusion is the pinned notebook's: decide_scores and the native_stage line are executed from
the pinned vendor/infer.py source and compared on random inputs; OOF targets are deterministic and sane."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared import BASE
import proxy_v3 as X
import oof_targets as T


def pinned_functions():
    source = (BASE / 'vendor/infer.py').read_text()
    block = re.search(r'\ndef decide_scores\(P\):\n(?:    .*\n)+', source).group(0)
    soft = re.search(r'\ndef rowsoftmax\(z\):\n(?:    .*\n)+', source).group(0)
    line = 'probs += rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i]))) / len(folds)'
    assert source.count(line) == 1, 'pinned native_stage fusion line changed'
    assert source.count("p.add_argument('--sk_div', type=int, default=1)") == 1
    return block, soft


def test():
    block, soft = pinned_functions()
    rng = np.random.default_rng(3)
    y = rng.integers(0, 18, 400)
    scope = dict(np=np, a=SimpleNamespace(sinkhorn=X.SINKHORN, sinkhorn_target_mix=X.TARGET_MIX), y=y)
    exec(block + soft, scope)
    P = X.rowsoftmax(rng.normal(size=(300, 18)) * 2)
    assert np.allclose(scope['decide_scores'](P), X.decide_scores(P, X.label_prior(y)))
    parent = X.lsm(rng.normal(size=(300, 18)) * 3)
    native = X.lsm(rng.normal(size=(300, 18)))
    share = X.NATIVE_SHARE
    logP = np.log(np.maximum(np.exp(parent), 1e-15))
    pinned = scope['rowsoftmax'](scope['decide_scores'](scope['rowsoftmax']((1 - share) * logP + share * native)))
    assert np.allclose(pinned, X.deploy(parent, native, X.label_prior(y)))
    # Notebook settings the proxy mirrors.
    cell = json.loads((BASE / 'external/pinned_notebook/ntl-684-ozertuu-mirror.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in cell['cells'])
    for flag, value in (('--native_share', '0.40'), ('--sinkhorn', '0.30'), ('--sinkhorn_target_mix', '0.35')):
        assert f"'{flag}', '{value}'" in text, flag
    weights = [float(w) for w in re.findall(r'"w": ([0-9.]+)', text)]
    assert abs(sum(weights) / len(weights) - X.PARENT_TEMPERATURE) < 1e-9, weights
    # Groups never exceed the cap and stay within one prompt.
    prompts = np.repeat(np.arange(20), 7)
    g = X.test_like_groups(prompts, 2, 0)
    assert np.bincount(g).max() <= 2 and all(len(set(prompts[g == q])) == 1 for q in np.unique(g))
    # Rebuild the targets in memory: identical to the recorded cache when present.
    payload, record = T.build()
    assert payload['weight'].min() >= T.EPS - 1e-12 and payload['weight'].max() <= 1 + 1e-12
    assert np.allclose(payload['weight'], T.weight_of(payload['p_true']))
    import common as C
    assert np.allclose(X.lsm(payload['parent_lp'].astype(np.float64) / X.PARENT_TEMPERATURE), C.public_lora_stack(), atol=1e-4)
    if T.DEFAULT.exists():
        meta = json.loads(T.DEFAULT.with_suffix('.json').read_text())
        with np.load(T.DEFAULT) as z:
            assert np.array_equal(z['weight'], payload['weight']) and meta['inputs'] == record['inputs']
    print(json.dumps(dict(decide_scores_matches_pinned=True, native_stage_line_matches_pinned=True,
                          notebook_flags_match=True, parent_temperature=X.PARENT_TEMPERATURE,
                          targets_deterministic=True, **record['summary'])))


if __name__ == '__main__':
    test()
