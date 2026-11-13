"""SAGE (B): sibling-aware gated evidence re-ranker. numpy only; this exact file is fitted offline and embedded in
the Kaggle notebook, so the runtime features are the validated ones.

For each row, take the top-K candidates of the final stack probabilities (public stack + A1 native). A row is
GATED when its top-2 are siblings / fixed confusables, or its top-1 margin is in the lower GATE_Q quantile of the
same dataset. Gated rows are re-ranked by a conditional logit over their K candidates; other rows keep top-1.
Candidate features (relative within the row, so they do not depend on the absolute calibration of the stack):
  rel_final (z-scored on the dataset at hand: transductive standardization, because the real stack is sharper
  than any offline proxy), rank indicators, rel_typo (length-conditioned typography likelihood estimated on
  external data only), neighbor-of-top-1, and the A2 CCL-Gen evidence (sum log-likelihood difference /
  sqrt(tokens)) alone and interacted with neighbor-of-top-1. Rows without A2 scores use the no-A2 weights.
There is deliberately NO native feature: the native is already inside the stack, and re-learning the stack/native
trade-off on an offline proxy only reproduces the proxy's miscalibration (+2.63 BA at proxy share 0.40, -0.30 once
the proxy is LB-anchored). B can only act on evidence the stack does not have.
"""
import json
import re

import numpy as np

K = 4
GATE_Q = .5
LEN_BINS = (80, 110, 150)
DASH = ((re.compile(r' — '), 2), (re.compile(r'—'), 1), (re.compile(r'–'), 3), (re.compile(r' - '), 4))
EXTRAS = (re.compile(r';'), re.compile(r':'), re.compile(r'\('))
N_JOINT = 5 * 2 * 2 * 2
KAPPA = 20.
FEATURES_A2 = ['rel_final', 'rank2', 'rank3', 'rank4', 'rel_typo', 'neighbor', 'rel_a2', 'rel_a2_x_neighbor']
FEATURES_NO_A2 = FEATURES_A2[:6]


# ---------- typography evidence (estimated on external data only)
def typography(texts):
    joint, extras, lbin = [], [], []
    for t in texts:
        t = str(t)
        dash = next((code for rx, code in DASH if rx.search(t)), 0)
        joint.append(((dash * 2 + ('“' in t or '”' in t)) * 2 + ('"' in t)) * 2 + ('’' in t))
        extras.append([bool(rx.search(t)) for rx in EXTRAS])
        lbin.append(int(np.searchsorted(LEN_BINS, len(t.split()), side='right')))
    return np.asarray(joint), np.asarray(extras, float), np.asarray(lbin)


def fit_typography(texts, labels, n_classes):
    joint, extras, lbin = typography(texts)
    nb = len(LEN_BINS) + 1
    J = np.zeros((n_classes, nb, N_JOINT)); X = np.zeros((n_classes, nb, len(EXTRAS))); C = np.zeros((n_classes, nb))
    np.add.at(J, (labels, lbin, joint), 1); np.add.at(X, (labels, lbin), extras); np.add.at(C, (labels, lbin), 1)
    back = (J.sum(1) + 1) / (J.sum((1, 2))[:, None] + N_JOINT)
    pj = (J + KAPPA * back[:, None, :]) / (C[..., None] + KAPPA)
    px = (X + 1) / (C[..., None] + 2)
    return dict(log_joint=np.log(pj).tolist(), log_extra_on=np.log(px).tolist(), log_extra_off=np.log(1 - px).tolist())


def typography_lp(table, texts):
    joint, extras, lbin = typography(texts)
    lj = np.asarray(table['log_joint']); on = np.asarray(table['log_extra_on']); off = np.asarray(table['log_extra_off'])
    out = lj[:, lbin, joint].T
    out += np.einsum('nk,cnk->nc', extras, on[:, lbin]) + np.einsum('nk,cnk->nc', 1 - extras, off[:, lbin])
    return out


# ---------- candidates, gate, features
def lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def candidates(final_probs, k=K):
    return np.argsort(-final_probs, 1, kind='stable')[:, :k]


def gate(final_probs, cand, neighbors, gate_q=GATE_Q):
    lp = np.log(np.maximum(final_probs, 1e-15))
    rows = np.arange(len(cand))
    margin = lp[rows, cand[:, 0]] - lp[rows, cand[:, 1]]
    quantile = (np.argsort(np.argsort(margin, kind='stable'), kind='stable') + .5) / len(margin)
    return np.asarray(neighbors)[cand[:, 0], cand[:, 1]] | (quantile < gate_q)


def features(final_probs, typo_lp, cand, neighbors, a2_sums=None, a2_tokens=None, scale=None):
    """X [N, K, D]; a2_sums aligned with cand ([N, K]) or None. scale: std of rel_final on this dataset."""
    rows = np.arange(len(cand))[:, None]
    lp = np.log(np.maximum(final_probs, 1e-15))
    rel_final = lp[rows, cand] - lp[rows, cand[:, :1]]
    if scale is None:
        scale = float(rel_final[:, 1:].std()) or 1.
    typo = typo_lp[rows, cand]
    neighbor = np.asarray(neighbors)[cand[:, :1], cand].astype(float)
    k = cand.shape[1]
    ranks = np.zeros((len(cand), k, 3)); ranks[:, 1, 0] = 1; ranks[:, 2, 1] = 1
    if k > 3:
        ranks[:, 3:, 2] = 1
    cols = [rel_final / scale, ranks[..., 0], ranks[..., 1], ranks[..., 2], typo - typo.mean(1, keepdims=True), neighbor]
    if a2_sums is not None:
        rel = (a2_sums - a2_sums.max(1, keepdims=True)) / np.sqrt(np.maximum(a2_tokens, 1))[:, None]
        rel = np.clip(rel, -20, 0)
        cols += [rel, rel * neighbor]
    return np.stack(cols, -1), scale


# ---------- conditional logit
def fit_logit(X, target, C):
    from scipy.optimize import minimize
    D = X.shape[-1]

    def f(w):
        s = X @ w; ls = lsm(s)
        nll = -ls[np.arange(len(target)), target].sum()
        p = np.exp(ls)
        grad = -(X[np.arange(len(target)), target] - (p[..., None] * X).sum(1)).sum(0)
        return nll + .5 * w @ w / C, grad + w / C
    return minimize(f, np.zeros(D), jac=True, method='L-BFGS-B', options=dict(maxiter=1000, ftol=1e-13, gtol=1e-9)).x


def decide(X, cand, gated, w):
    out = cand[:, 0].copy()
    if gated.any():
        out[gated] = cand[gated, np.argmax(X[gated] @ w, 1)]
    return out


def training_target(cand, gated, y):
    hit = (cand == y[:, None])
    use = gated & hit.any(1)
    return use, hit.argmax(1)


# ---------- runtime (Kaggle): A2 workers on both GPUs, then B
def run(probs, test, query, log, write_pred, labels, bundle_b, bundle_a2, base_a2, ccl_script, work, deadline):
    import os, subprocess, sys, time
    B = json.load(open(bundle_b))
    if list(labels) != list(B['classes']) or probs.shape[1] != len(labels):
        raise ValueError('Runtime label order differs from the fitted B bundle')
    neighbors = np.asarray(B['neighbors'], bool)
    cand = candidates(probs, B['k'])
    gated = gate(probs, cand, neighbors, B['gate_q'])
    typo = typography_lp(B['typography'], test.text.fillna('').astype(str).values)
    os.makedirs(work, exist_ok=True)
    np.save(os.path.join(work, 'cand.npy'), cand)
    sums = np.zeros(cand.shape); tokens = np.zeros(len(cand)); done = np.zeros(len(cand), bool)
    jobs = []
    try:
        import torch
        n_gpu = max(1, torch.cuda.device_count())
        rows_all = np.flatnonzero(gated)
        for g in range(n_gpu):
            rows = rows_all[g::n_gpu].tolist()
            if not rows:
                continue
            req = os.path.join(work, f'req{g}.json')
            json.dump(dict(bundle=bundle_a2, base=base_a2, query=query, candidates=os.path.join(work, 'cand.npy'),
                           rows=rows, out=os.path.join(work, f'out{g}.npz'), deadline=deadline, batch=4), open(req, 'w'))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            jobs.append((subprocess.Popen([sys.executable, ccl_script, req], env=env), os.path.join(work, f'out{g}.npz')))
        for proc, out in jobs:
            try:
                proc.wait(timeout=max(60., deadline + 300 - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); log('DYL A2 worker killed after its deadline')
            if proc.returncode == 0 and os.path.exists(out):
                z = np.load(out)
                sums[z['rows']] = z['sums']; tokens[z['rows']] = z['tokens']; done[z['rows']] = z['done']
    except Exception as exc:
        log('DYL A2 failed; gated rows use the no-A2 re-ranker:', repr(exc))
    X0, s0 = features(probs, typo, cand, neighbors)
    final = decide(X0, cand, gated, np.asarray(B['w_no_a2']))
    with_a2 = gated & done
    if with_a2.any():
        X1, _ = features(probs, typo, cand, neighbors, sums, tokens, scale=s0)
        final[with_a2] = decide(X1[with_a2], cand[with_a2], np.ones(int(with_a2.sum()), bool), np.asarray(B['w_a2']))
    log(f'DYL_V4_SAGE gated {int(gated.sum())}/{len(cand)}, A2 scored {int(done.sum())}, '
        f'changed vs stack {int((final != cand[:, 0]).sum())}')
    write_pred(final, 'dyl v4 A1+A2+B')
    return final
