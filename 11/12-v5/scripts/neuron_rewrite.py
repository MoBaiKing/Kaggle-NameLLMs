"""SNR — Sibling Neuron Rewrite of the public native expert's output neurons (CPU; no model forward needed).

Both native heads are linear (18 output neurons over the 32,768-d multi-depth feature). Rewriting the neurons as
    W' = (I + D) W,    b' = (I + D) b + d,
with D supported only on the diagonal and on sibling / fixed-confusable pairs (families.NEIGHBORS: the 61 % of
public-rule errors stay inside a family), changes the head outputs exactly to  z' = (I + D) z + d  for every text,
window and fold state. The author's clean OOF (oof_reference.npz) holds z for every official row from the fold
state that never saw it, so D and d are fitted on clean held-out logits and then written into all five fold files;
the public native_q8 runtime loads them unchanged.

Fitting: L-BFGS on the mean cross-entropy of the native's own calibrated output (public plain/centered scales and
centred weight frozen; centred head on hidden-test-like groups of <= 2 answers per prompt, several draws) plus
lam * (||D||^2 + ||d||^2)  (ODIR-style regularisation of Dirichlet calibration, Kull et al., NeurIPS 2019; here
restricted to the sibling structure). The objective is the native expert alone, deliberately not the proxy
ensemble, so the fit cannot exploit the proxy parent's miscalibration. lam is chosen by inner prompt-fold CV.
"""
import json
from pathlib import Path

import numpy as np
import torch

import shared  # noqa: F401
import common as C
import families as F

LAMS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
FIT_DRAWS = 4
GROUP_MAX = 2


def load_oof():
    pp, pc, nfold = C.public_native_oof()
    states = [C.public_state(k) for k in range(5)]
    bias = np.stack([C.public_bias(k) for k in range(5)])
    return dict(plain=pp, centered=pc, fold=nfold, bias=bias,
                ps=np.array([s['plain_scale'] for s in states]), cs=np.array([s['centered_scale'] for s in states]),
                cw=np.array([s['selected_centered_weight'] for s in states]), y=C.y_all, pid=C.pid)


def groups_m(prompts, m, seed):
    rng = np.random.default_rng(seed)
    g = np.zeros(len(prompts), np.int64); nxt = 0
    for q in np.unique(prompts):
        r = rng.permutation(np.flatnonzero(prompts == q))
        for i in range(0, len(r), m):
            g[r[i:i + m]] = nxt; nxt += 1
    return g


def relative(centered, groups, bias_rows):
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    means = np.zeros((len(cnt), centered.shape[1])); np.add.at(means, inv, centered); means /= cnt[:, None]
    return centered - means[inv] + bias_rows, cnt[inv] >= 2


MASK = np.eye(18, dtype=bool) | F.NEIGHBORS


def fit_head(z, scale, y, lam, mask=MASK):
    """z [n,18] head outputs (bias included), scale [n] public temperature -> (D, d)."""
    Z = torch.tensor(z, dtype=torch.float64); S = torch.tensor(scale, dtype=torch.float64)[:, None]
    Y = torch.tensor(y); M = torch.tensor(mask, dtype=torch.float64)
    D = torch.zeros(18, 18, dtype=torch.float64, requires_grad=True); d = torch.zeros(18, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([D, d], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        Dm = D * M
        out = (Z @ (torch.eye(18, dtype=torch.float64) + Dm).T + d) * S
        loss = torch.nn.functional.cross_entropy(out, Y) + lam * ((Dm ** 2).sum() + (d ** 2).sum())
        loss.backward()
        return loss
    opt.step(closure)
    return (D * M).detach().numpy(), d.detach().numpy()


def apply(z, D, d):
    return z @ (np.eye(18) + D).T + d


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def native_lp(o, rows, edit, groups):
    """Calibrated native log-probs (public native_q8.calibrate semantics) for rows, optionally with an edit."""
    k = o['fold'][rows]
    plain, cent = o['plain'][rows], o['centered'][rows]
    rel, elig = relative(cent, groups[rows], o['bias'][k])
    if edit is not None:
        plain = apply(plain, *edit['plain']); rel = apply(rel, *edit['centered'])
    a = lsm(plain * o['ps'][k][:, None]); b = lsm(rel * o['cs'][k][:, None])
    f = np.where(elig, o['cw'][k], 0.)[:, None]
    return lsm((1 - f) * a + f * b)


def entropy_match(z, scale, edit_head, target_entropy):
    """Scalar s with mean entropy of softmax(s * scale * ((I+D) z + d)) equal to target (bisection)."""
    D, d = edit_head
    out = apply(z, D, d) * scale[:, None]
    ent = lambda t: float(-(np.exp(lsm(t * out)) * lsm(t * out)).sum(1).mean())
    lo, hi = .2, 5.
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ent(mid) > target_entropy else (lo, mid)
    t = (lo + hi) / 2
    return (np.eye(18) + D) * t - np.eye(18), d * t


def fit(o, rows, lam, seed0=0, match_entropy=False):
    """Fit plain on all rows; centred on eligible rows of FIT_DRAWS test-like groupings."""
    k = o['fold'][rows]
    plain = fit_head(o['plain'][rows], o['ps'][k], o['y'][rows], lam)
    rels, ys, scales = [], [], []
    for draw in range(FIT_DRAWS):
        g = groups_m(o['pid'], GROUP_MAX, 500 + seed0 * 10 + draw)
        rel, elig = relative(o['centered'][rows], g[rows], o['bias'][k])
        rels.append(rel[elig]); ys.append(o['y'][rows][elig]); scales.append(o['cs'][k][elig])
    cent = fit_head(np.concatenate(rels), np.concatenate(scales), np.concatenate(ys), lam)
    if match_entropy:
        z, sc = o['plain'][rows], o['ps'][k]
        plain = entropy_match(z, sc, plain, float(-(np.exp(lsm(z * sc[:, None])) * lsm(z * sc[:, None])).sum(1).mean()))
        R, SC = np.concatenate(rels), np.concatenate(scales)
        cent = entropy_match(R, SC, cent, float(-(np.exp(lsm(R * SC[:, None])) * lsm(R * SC[:, None])).sum(1).mean()))
    return dict(plain=plain, centered=cent)


def nll_rows(o, rows, edit, draws=3):
    out = []
    for draw in range(draws):
        g = groups_m(o['pid'], GROUP_MAX, 900 + draw)
        lp = native_lp(o, rows, edit, g)
        out.append(-lp[np.arange(len(rows)), o['y'][rows]].mean())
    return float(np.mean(out))


def choose_lam(o, rows):
    scores = {}
    for lam in LAMS:
        total = 0.
        for j in np.unique(o['fold'][rows]):
            tr, te = rows[o['fold'][rows] != j], rows[o['fold'][rows] == j]
            total += nll_rows(o, te, fit(o, tr, lam, j)) * len(te)
        scores[lam] = total / len(rows)
    return min(scores, key=scores.get), scores


def nested_edits(o, match_entropy=False):
    """Outer prompt folds: for fold k, lam by inner CV on the other folds, fit there, apply to fold k."""
    out = {}
    for k in range(5):
        rows = np.flatnonzero(o['fold'] != k)
        lam, scores = choose_lam(o, rows)
        out[k] = dict(edit=fit(o, rows, lam, 50 + k, match_entropy), lam=lam, inner=scores)
    return out


def write_heads(edit, public_file, out_file):
    """Rewrite the four head tensors of one public fold state; every other tensor is copied byte-for-byte."""
    from safetensors.numpy import load_file, save_file
    t = load_file(str(public_file))
    for head in ('plain', 'centered'):
        D, d = edit[head]
        A = np.eye(18) + D
        W, b = t[f'{head}.weight'].astype(np.float64), t[f'{head}.bias'].astype(np.float64)
        t[f'{head}.weight'] = (A @ W).astype(np.float32)
        t[f'{head}.bias'] = (A @ b + d).astype(np.float32)
    tmp = str(out_file) + '.tmp'
    save_file(t, tmp)
    Path(tmp).replace(out_file)


def to_json(edit):
    return {h: dict(D=edit[h][0].tolist(), d=edit[h][1].tolist()) for h in ('plain', 'centered')}


def from_json(obj):
    return {h: (np.asarray(obj[h]['D']), np.asarray(obj[h]['d'])) for h in ('plain', 'centered')}
