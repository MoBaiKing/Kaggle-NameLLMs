"""Rebuild the frozen weighted stream from its exact public source snapshot."""
import argparse, hashlib, json
import numpy as np, pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = str(ROOT) + '/'
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--cap', type=int, required=True)
ap.add_argument('--pool', required=True, help='Exact original Open Pool Parquet snapshot')
ap.add_argument('--out', default=str(ROOT / 'data/stream_8b_cap9500.parquet'))
ap.add_argument('--seed', type=int, default=20261110); ap.add_argument('--clip-lo', type=float, default=0.1); ap.add_argument('--clip-hi', type=float, default=3.0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
tr = pd.read_csv(ROOT / 'data/original/train.csv', dtype=str); classes = sorted(tr.model.unique())
# Reconstruct the frozen exclusion set from source rows; no reinforcement learning is run.
raw = pd.read_parquet(a.pool, columns=['text', 'model', 'prompt_group'])
raw = raw[raw.model.isin(classes) & ~raw.text.isin(set(tr.text.values))]
used = raw.sample(frac=1.0, random_state=42).groupby('model', group_keys=False).head(int(np.ceil(15000 / len(classes))))
rest = raw[~raw.text.isin(set(used.text))].drop_duplicates('text')
lo, hi = np.percentile(tr.text.str.split().str.len(), [1, 99])
words = rest.text.str.split().str.len()
rest = rest[(words >= lo) & (words <= hi)]
rl = rest.sample(frac=1.0, random_state=7).groupby('model', group_keys=False).head(1000).reset_index(drop=True)
expected = json.load(open(ROOT / 'config/assets.json'))['augmentation']['raw_pool_sha256']
assert hashlib.sha256(Path(a.pool).read_bytes()).hexdigest() == expected, 'Raw pool snapshot changed; use the frozen Release or locate the exact historical snapshot'

p = pd.read_parquet(a.pool,
                    columns=['text', 'model', 'prompt_canon', 'match_weight', 'source_dataset', 'n_words', 'flag_near_train_prompt', 'flag_train_prompt_regen'])
n0 = len(p)
p = p[p.model.isin(classes)]
p = p[~p.text.isin(set(tr.text)) & ~p.text.isin(set(rl.text))]
p = p[~p.flag_near_train_prompt & ~p.flag_train_prompt_regen]          # no official-prompt look-alikes (fold-0 val stays clean)
p = p.sort_values('match_weight', ascending=False, kind='stable').drop_duplicates('text').sort_index().reset_index(drop=True)
p['wt'] = p.match_weight.clip(a.clip_lo, a.clip_hi)
keep, resid, rep = [], [], {}
for c, g in p.groupby('model', sort=True):
    w = g.wt.values; n = len(g); k = min(n, a.cap)
    if k < n:
        key = rng.random(n) ** (1.0 / w); sel = np.argsort(-key)[:k]
        lo, hi = 0.0, 1.0 / w.min()                                   # tau with sum(min(1, tau*w)) = k
        for _ in range(80):
            tau = (lo + hi) / 2
            lo, hi = (tau, hi) if np.minimum(1, tau * w).sum() < k else (lo, tau)
        pi = np.minimum(1, tau * w)
    else:
        sel = np.arange(n); pi = np.ones(n)
    r = (w / pi)[sel]; r = r / r.mean()
    keep.append(g.index.values[sel]); resid.append(r)
    rep[c] = dict(candidates=int(n), selected=int(k), mean_match_weight_candidates=round(float(g.match_weight.mean()), 3),
                  mean_match_weight_selected=round(float(g.match_weight.values[sel].mean()), 3),
                  loss_weight_min=round(float(r.min()), 3), loss_weight_max=round(float(r.max()), 3),
                  effective_n=int(r.sum() ** 2 / (r ** 2).sum()))
idx = np.concatenate(keep); out = p.loc[idx, ['text', 'model', 'prompt_canon', 'match_weight', 'source_dataset', 'n_words']].copy()
out['w'] = np.concatenate(resid).astype(np.float32)
out['prompt'] = ['pc:%d' % i for i in pd.factorize(out.prompt_canon)[0]]
out = out.drop(columns='prompt_canon').sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
assert not out.text.isin(set(tr.text)).any() and not out.text.isin(set(rl.text)).any() and out.text.is_unique
f = a.out; Path(f).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(f, index=False)
g = out.groupby('prompt'); nm = g.model.nunique()
report = dict(tag=a.tag, cap=a.cap, seed=a.seed, clip=[a.clip_lo, a.clip_hi], pool_rows=int(n0), candidates=int(len(p)), selected=int(len(out)),
              prompts=int(len(nm)), rows_in_prompts_with_ge2_models=int(g.size()[nm >= 2].sum()), rows_in_prompts_with_ge4_models=int(g.size()[nm >= 4].sum()),
              words=dict(mean=round(float(out.n_words.mean()), 1), p05=int(out.n_words.quantile(.05)), p95=int(out.n_words.quantile(.95))),
              per_class=rep, by_source_top=out.source_dataset.value_counts().head(15).to_dict(),
              sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest())
json.dump(report, open(str(Path(f).with_suffix('.report.json')), 'w'), indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ('per_class', 'by_source_top')}))
print(pd.DataFrame(rep).T.to_string())

expected_stream = json.load(open(ROOT / 'config/assets.json'))['augmentation']['sha256']
assert report['sha256'] == expected_stream, 'Stream bytes differ: verify raw snapshot, package versions, parameters, and row order'
