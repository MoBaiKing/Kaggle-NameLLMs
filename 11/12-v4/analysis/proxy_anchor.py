import sys
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import shared, common as C, proxy_v3 as X, oof_targets as T
parent, _, _ = T.load_verified(shared.V3 / 'cache/oof_targets.npz', shared.BASE / 'data/original/train.csv', shared.BASE / 'data/folds.json', C.N)
prior = X.label_prior(C.y_all); y = C.y_all
states = {k: C.public_state(k) for k in range(5)}; biases = {k: C.public_bias(k) for k in range(5)}
row_fold = np.full(C.N, -1)
for k, r in C.FOLDS.items(): row_fold[r] = k
pp, pc, _ = C.public_native_oof()
natives = [X.native_rows(C.native_q8.calibrate, pp, pc, row_fold, biases, states, X.test_like_groups(C.pid, 2, 2000 + d)) for d in range(X.DRAWS)]
res = {}
for share in np.round(np.arange(.6, .96, .025), 3):
    fin = sum(X.deploy(parent, n, prior, share=share) for n in natives) / X.DRAWS
    res[float(share)] = 100 * X.ba(y, fin.argmax(1))
print({k: round(v, 2) for k, v in res.items()})
# smoothed optimum (3-point moving average)
ks = sorted(res); sm = {ks[i]: np.mean([res[ks[j]] for j in range(max(0, i - 1), min(len(ks), i + 2))]) for i in range(len(ks))}
print('smoothed argmax', max(sm, key=sm.get), round(max(sm.values()), 2))
