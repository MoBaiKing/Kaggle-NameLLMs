"""CPU analysis of saved clean OOF only: where does the public 0.684 rule fail? (validation proxy)"""
import sys, json
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import shared  # noqa
import common as C
import proxy_v3 as X
import oof_targets as T
parent, w, _ = T.load_verified(shared.V3 / 'cache/oof_targets.npz', shared.BASE / 'data/original/train.csv', shared.BASE / 'data/folds.json', C.N)
pp, pc, rf = C.public_native_oof()
st = {k: C.public_state(k) for k in range(5)}; bi = {k: C.public_bias(k) for k in range(5)}
prior = X.label_prior(C.y_all); y = C.y_all; cls = C.classes
fin = np.zeros((C.N, 18)); nat = np.zeros((C.N, 18))
for d in range(10):
    g = X.test_like_groups(C.pid, 2, 1000 + d)
    nl = X.native_rows(C.native_q8.calibrate, pp, pc, rf, bi, st, g)
    fin += X.deploy(parent, nl, prior) / 10; nat += np.exp(nl) / 10
fam = lambda c: c.split('-')[0] if not c.startswith('qwen') else 'qwen'
F = np.array([fam(c) for c in cls]); fy = F[y]
out = {}
for name, P in [('final', fin), ('native_alone', nat), ('lora_parent', np.exp(parent))]:
    pr = P.argmax(1); rank = (-P).argsort(1); pos = np.array([np.where(rank[i] == y[i])[0][0] for i in range(C.N)])
    err = pr != y
    out[name] = dict(ba=round(100 * X.ba(y, pr), 2), acc=round(100 * (~err).mean(), 2),
                     topk_recall={k: round(100 * (pos < k).mean(), 2) for k in (1, 2, 3, 5)},
                     family_acc=round(100 * (F[pr] == fy).mean(), 2),
                     errors=int(err.sum()), within_family_error_share=round(100 * (F[pr][err] == fy[err]).mean(), 1))
out['oracle_native_or_parent_correct'] = round(100 * ((nat.argmax(1) == y) | (parent.argmax(1) == y)).mean(), 2)
out['agreement_native_parent'] = round(100 * (nat.argmax(1) == parent.argmax(1)).mean(), 2)
pr = fin.argmax(1)
out['per_class_recall'] = {c: round(100 * (pr[y == i] == i).mean(), 1) for i, c in enumerate(cls)}
conf = {}
for t_, p_ in zip(y[pr != y], pr[pr != y]):
    conf[(cls[t_], cls[p_])] = conf.get((cls[t_], cls[p_]), 0) + 1
out['top_confusions'] = [f'{a} -> {b}: {n}' for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])[:15]]
srt = np.sort(fin, 1); margin = srt[:, -1] - srt[:, -2]
out['acc_by_margin_quintile'] = [round(100 * (pr[(margin >= lo) & (margin <= hi)] == y[(margin >= lo) & (margin <= hi)]).mean(), 1)
                                 for lo, hi in zip(np.quantile(margin, [0, .2, .4, .6, .8]), np.quantile(margin, [.2, .4, .6, .8, 1]))]
fam_err = {}
for f in sorted(set(F)):
    m = (fy == f) & (pr != y)
    fam_err[f] = dict(rows=int((fy == f).sum()), errors=int(m.sum()), within=int((F[pr][m] == f).sum()))
out['errors_by_true_family'] = fam_err
print(json.dumps(out, indent=1))
