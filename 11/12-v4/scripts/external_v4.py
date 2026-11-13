"""CPU only: the external training sets for A1 (pool replay) and A2 (CCL-Gen) from the pinned 11/11 stream
(Open Pool selection, already typography-matched to the official texts: 0% line breaks / bold).

Prompt-clean filter: a stream row is dropped if its word TF-IDF similarity to any official prompt centroid is
>= SIM_MAX (0.30 ~ the 99th percentile of an official answer's similarity to OTHER official prompts), so the
2,699 official rows stay clean held-out data for A2 (never trained on, no shared prompt topic).
A2 split: a dev set of whole stream prompt clusters (monitoring only, never used for selection).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared import BASE, ROOT
from oof_targets import sha
import families as F

SIM_MAX = .30
A2_CAP = 3200            # rows per class for A2 training (smaller classes keep all)
A2_DEV_PER_CLASS = 60
SEED = 20261113
DEFAULT = ROOT / 'cache/external_v4.parquet'


def prompt_similarity(official, stream_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(stop_words='english', min_df=2, max_features=200000, sublinear_tf=True)
    vec.fit(pd.concat([official.text, stream_text.sample(min(40000, len(stream_text)), random_state=0)]))
    T = normalize(vec.transform(official.text))
    S = normalize(vec.transform(stream_text))
    prompts = official.prompt_id.values
    centroids = normalize(np.vstack([np.asarray(T[prompts == p].mean(0)) for p in np.unique(prompts)]))
    sim = np.zeros(len(stream_text))
    for i in range(0, len(stream_text), 20000):
        sim[i:i + 20000] = (S[i:i + 20000] @ centroids.T).max(1)
    return sim


def build(stream_path, official_path):
    stream = pd.read_parquet(stream_path).reset_index(drop=True)
    official = pd.read_csv(official_path, dtype=str)
    if sorted(stream.model.unique()) != F.CLASSES or sorted(official.model.unique()) != F.CLASSES:
        raise ValueError('Class set differs from the 18 competition labels')
    if set(stream.text) & set(official.text):
        raise ValueError('External rows repeat official texts')
    stream['stream_row'] = np.arange(len(stream))
    stream['prompt_sim'] = prompt_similarity(official, stream.text)
    clean = stream[stream.prompt_sim < SIM_MAX].copy()
    rng = np.random.default_rng(SEED)
    clusters = clean.prompt.astype(str).values
    unique = np.unique(clusters)
    dev_clusters = set(rng.choice(unique, size=max(1, len(unique) // 40), replace=False))
    clean['split'] = np.where(np.isin(clusters, list(dev_clusters)), 'dev_pool', 'train_pool')
    a2 = np.zeros(len(clean), bool); dev = np.zeros(len(clean), bool)
    for c in F.CLASSES:
        rows = np.flatnonzero((clean.model.values == c) & (clean.split.values == 'train_pool'))
        a2[rng.permutation(rows)[:A2_CAP]] = True
        rows = np.flatnonzero((clean.model.values == c) & (clean.split.values == 'dev_pool'))
        dev[rng.permutation(rows)[:A2_DEV_PER_CLASS]] = True
    clean['a2_train'] = a2
    clean['a2_dev'] = dev
    clean['label'] = np.searchsorted(F.CLASSES, clean.model.values)
    keep = ['stream_row', 'text', 'model', 'label', 'prompt', 'source_dataset', 'n_words', 'prompt_sim', 'split', 'a2_train', 'a2_dev']
    summary = dict(stream_rows=int(len(stream)), dropped_prompt_overlap=int(len(stream) - len(clean)), kept=int(len(clean)),
                   a2_train=int(a2.sum()), a2_dev=int(dev.sum()),
                   a2_train_per_class={c: int(((clean.model.values == c) & a2).sum()) for c in F.CLASSES})
    return clean[keep].reset_index(drop=True), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, default=BASE / 'data/stream_8b_cap9500.parquet')
    p.add_argument('--out', type=Path, default=DEFAULT)
    a = p.parse_args()
    spec = json.loads((BASE / 'config/assets.json').read_text())
    if sha(a.stream) != spec['augmentation']['sha256']:
        raise SystemExit('Pinned external stream changed')
    official = BASE / 'data/original/train.csv'
    frame, summary = build(a.stream, official)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_name(a.out.stem + '.tmp.parquet'); frame.to_parquet(temp, index=False); temp.replace(a.out)
    record = dict(summary, stream_sha256=sha(a.stream), official_sha256=sha(official), sim_max=SIM_MAX, a2_cap=A2_CAP,
                  seed=SEED, payload_sha256=sha(a.out), code_sha256=sha(Path(__file__)))
    a.out.with_suffix('.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
