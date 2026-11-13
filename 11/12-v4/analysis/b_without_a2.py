"""CPU, saved OOF only: B (SAGE) with its final feature set (no native feature) and WITHOUT A2, under nested
prompt-fold CV, on the public-native stack proxy at native share 0.40 (unanchored) and 0.75 (LB-anchored).
Expected: no real gain either way, i.e. B's value can only come from A2's new evidence. (The earlier B variant
with a native feature gained +2.63 at 0.40 and -0.30 once anchored; that feature was removed.)"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'scripts'))
import json
import numpy as np, pandas as pd
import shared, common as C, proxy_v3 as X, oof_targets as T, families as F, sage as S, external_v4 as E
import decide_v4 as D
import proxy_v4 as X4
parent, _, _ = T.load_verified(shared.V3 / 'cache/oof_targets.npz', shared.BASE / 'data/original/train.csv', shared.BASE / 'data/folds.json', C.N)
prior = X.label_prior(C.y_all); y = C.y_all
states = {k: C.public_state(k) for k in range(5)}; biases = {k: C.public_bias(k) for k in range(5)}
row_fold = np.full(C.N, -1)
for k, r in C.FOLDS.items():
    row_fold[r] = k
pp, pc, _ = C.public_native_oof()
ext = pd.read_parquet(E.DEFAULT); ext = ext[ext.split == 'train_pool']
typo = S.typography_lp(S.fit_typography(ext.text.values, ext.label.values, 18), C.tr.text.fillna('').astype(str).values)
out = {}
for share in (.40, X4.ANCHOR_SHARE):
    original = X4.deploy
    X4.deploy = lambda parent, native, prior, share=share, **kw: X.deploy(parent, native, prior, share=share, **kw)
    s0 = D.stage_probs(C, parent, prior, pp, pc, biases, states, row_fold)
    X4.deploy = original
    pred, gated, chosen = D.nested(y, row_fold, s0, typo, None, F.NEIGHBORS)
    r = X.paired_report(y, C.pid, s0, D.onehot(pred))
    out[f'share_{share}'] = dict(stack_ba=round(r['base_ba'], 2), with_b=round(r['ba'], 2), delta=round(r['delta'], 2),
                                 ci95=[round(v, 2) for v in r['ci95']], fixed=r['fixed'], harmed=r['harmed'], gated=int(gated.sum()))
print(json.dumps(out, indent=1))
