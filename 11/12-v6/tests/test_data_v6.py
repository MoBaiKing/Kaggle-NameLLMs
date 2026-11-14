"""data_v6 rules on a synthetic v15-shaped frame: prompt-near rows dropped, w = clipped match_weight x fidelity factor
(only for kimi-k3 / glm-5.3 / qwen3.8-max, relative to the best setting, floored), official overlap refused."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import data_v6 as DV

CLASSES = sorted(['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview',
                  'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna',
                  'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'])


def test():
    rows = []
    for c in CLASSES:
        for j, g in enumerate(['high', 'off', 'default', 'unknown']):
            rows.append(dict(text=f'{c} {j}', model=c, prompt_group=f'g{j}', prompt_canon=None if j else f'canon{j}',
                             match_weight=[.01, 1., 2., 80.][j], gen_reasoning=g, max_sim_train_prompt=[.1, .1, .3, .1][j],
                             id=f'{c}{j}', source_dataset='s'))
    pool = pd.DataFrame(rows)
    fid = pd.DataFrame([dict(scope='all sources', model='kimi-k3', gen_reasoning='high', fidelity_ratio=.8),
                        dict(scope='all sources', model='kimi-k3', gen_reasoning='off', fidelity_ratio=.2),
                        dict(scope='all sources', model='kimi-k3', gen_reasoning='default', fidelity_ratio=.4)])
    official = pd.DataFrame(dict(text=[f'official {c}' for c in CLASSES], model=CLASSES, prompt_id='p'))
    out, summary = DV.build(pool, fid, official)
    assert summary['dropped_prompt_near'] == 18 and len(out) == 54
    k = out[out.model == 'kimi-k3'].set_index('gen_reasoning').w
    assert np.isclose(k['high'], DV.W_MIN) and np.isclose(k['off'], 1. * DV.FID_FLOOR) and np.isclose(k['unknown'], DV.W_MAX * DV.FID_FLOOR)
    g = out[out.model == 'gpt-5.5'].set_index('gen_reasoning').w
    assert np.isclose(g['off'], 1.) and np.isclose(g['unknown'], DV.W_MAX)
    assert out[out.model == 'gpt-5.5'].set_index('gen_reasoning').prompt['high'] == 'canon0'
    try:
        DV.build(pd.concat([pool, pd.DataFrame([dict(rows[0], text='official gpt-5.5')])]), fid, official)
    except ValueError:
        pass
    else:
        raise AssertionError('official overlap accepted')
    print(json.dumps(dict(prompt_near_dropped=True, weights_clipped=True, fidelity_floor=True, official_overlap_refused=True)))


if __name__ == '__main__':
    test()
