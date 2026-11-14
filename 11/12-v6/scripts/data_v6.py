"""CPU: the v6 external training stream from Open Pool v15 (Kaggle crtproduction/name-that-llm-open-pool, file
ntl_open_pool.parquet, SHA256 pinned below). Rules fixed before any v6 training, each backed by the dataset author's
measured ablations (dataset description, v11/v14 sections):
  * sampling weight w = match_weight (typography/length matched to the real texts; +0.026 balanced accuracy at
    equal size on held-out real prompts), clipped to [W_MIN, W_MAX];
  * generation-setting fidelity: for kimi-k3, glm-5.3 and qwen3.8-max the train-only recogniser shows that some
    reasoning settings look much more like the real passages (kimi-k3: high 25% vs off 8%). Rows are down-weighted
    by fidelity_ratio(model, setting) / best fidelity_ratio of that model, floored at FID_FLOOR (other labels: no
    consistent direction per the author -> factor 1);
  * prompt-clean: rows with max_sim_train_prompt >= SIM_MAX are dropped, so the official OOF gate stays clean;
  * class balance is done by the trainer's sampler (equal probability per label, within label proportional to w).
Output columns are the 11/12 stream layout (text, model, prompt, w) plus provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared import BASE, ROOT
from oof_targets import sha

POOL_SHA256 = '5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332'
POOL_ROWS = 226191
W_MIN, W_MAX = .05, 10.
FID_FLOOR = .3
FID_LABELS = ('kimi-k3', 'glm-5.3', 'qwen3.8-max')
SIM_MAX = .25
DEFAULT = ROOT / 'cache/stream_v6.parquet'


def fidelity_factors(table):
    """{(model, setting): factor} from fidelity_by_reasoning.csv rows with scope == 'all sources'."""
    t = table[(table.scope == 'all sources') & table.model.isin(FID_LABELS)]
    out = {}
    for model, g in t.groupby('model'):
        best = g.fidelity_ratio.max()
        for _, r in g.iterrows():
            out[(model, r.gen_reasoning)] = float(np.clip(r.fidelity_ratio / best, FID_FLOOR, 1.))
    return out


def build(pool, fidelity, official):
    classes = sorted(official.model.unique())
    if sorted(pool.model.unique()) != classes:
        raise ValueError('Pool labels differ from the 18 competition labels')
    if set(pool.text) & set(official.text):
        raise ValueError('Pool repeats official texts')
    keep = pool.max_sim_train_prompt.values < SIM_MAX
    p = pool[keep].copy()
    factors = fidelity_factors(fidelity)
    fac = np.array([factors.get((m, g), 1. if m not in FID_LABELS else FID_FLOOR) for m, g in zip(p.model, p.gen_reasoning)])
    p['w'] = np.clip(p.match_weight.values.astype(float), W_MIN, W_MAX) * fac
    p['prompt'] = p.prompt_group.astype(str).where(p.prompt_canon.isna(), p.prompt_canon.astype(str))
    cols = ['text', 'model', 'prompt', 'w', 'id', 'source_dataset', 'gen_reasoning', 'match_weight']
    out = p[cols].reset_index(drop=True)
    eff = out.groupby('model').w.apply(lambda w: float(w.sum() ** 2 / (w ** 2).sum()))
    summary = dict(pool_rows=int(len(pool)), dropped_prompt_near=int((~keep).sum()), kept=int(len(out)),
                   effective_rows_per_label={k: int(v) for k, v in eff.items()},
                   fidelity_factors={f'{m}|{g}': round(v, 3) for (m, g), v in sorted(factors.items())})
    return out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-dir', type=Path, default=BASE / 'external/open_pool_v15')
    p.add_argument('--out', type=Path, default=DEFAULT)
    a = p.parse_args()
    pool_file = a.pool_dir / 'ntl_open_pool.parquet'
    if sha(pool_file) != POOL_SHA256:
        raise SystemExit('Open Pool file differs from the pinned v15 bytes')
    pool = pd.read_parquet(pool_file)
    if len(pool) != POOL_ROWS:
        raise SystemExit('Unexpected Open Pool row count')
    official = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    out, summary = build(pool, pd.read_csv(a.pool_dir / 'fidelity_by_reasoning.csv'), official)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + '.tmp.parquet'); out.to_parquet(tmp, index=False); tmp.replace(a.out)
    record = dict(summary, pool_sha256=POOL_SHA256, fidelity_sha256=sha(a.pool_dir / 'fidelity_by_reasoning.csv'),
                  official_sha256=sha(BASE / 'data/original/train.csv'), payload_sha256=sha(a.out),
                  rules=dict(w_min=W_MIN, w_max=W_MAX, fid_floor=FID_FLOOR, fid_labels=FID_LABELS, sim_max=SIM_MAX),
                  code_sha256=sha(Path(__file__)))
    a.out.with_suffix('.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
