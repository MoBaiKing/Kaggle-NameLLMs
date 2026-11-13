"""Deployment-exact fusion of the pinned 0.684 notebook (infer.py native_stage), on clean OOF rows.

infer.py, per native fold state i:
    probs += rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i]))) / len(folds)
with share 0.40, Sinkhorn 0.30, target mix 0.35 and the official-train label prior. decide_scores below is a
verbatim copy (tests/test_proxy_v3.py checks it against the pinned vendor/infer.py). On OOF rows each row
has exactly one held-out native state, so the fold average degenerates to that state. VALIDATION PROXY ONLY.
"""
import numpy as np

NATIVE_SHARE = .40
SINKHORN = .30
TARGET_MIX = .35
# Public blend(): total = sum_i w_i * log p_i over 6 active members (weights 3, 2, 1, 2.5, 1, 1 = 10.5) and
# sk_div=1 divides by the 6 active members, i.e. 1.75 x the weighted mean log-probability.
PARENT_TEMPERATURE = 10.5 / 6
GROUP_MAX = 2
DRAWS = 10
CLASSES = 18


def lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def rowsoftmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def label_prior(y):
    prior = np.bincount(y, minlength=CLASSES).astype(np.float64) + 1.0
    return prior / prior.sum()


def decide_scores(P, prior, sinkhorn=SINKHORN, target_mix=TARGET_MIX):
    Q = P.copy()
    rho = float(np.clip(target_mix, 0.0, 1.0))
    marg = np.clip(P.mean(0), 1e-6, None)
    marg /= marg.sum()
    target = np.clip((1.0 - rho) * marg + rho * prior, 1e-6, None)
    target /= target.sum()
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - sinkhorn) * np.log(P + 1e-12) + sinkhorn * np.log(Q + 1e-12)


def deploy(parent_lp, native_lp, prior, share=NATIVE_SHARE, sinkhorn=SINKHORN, target_mix=TARGET_MIX):
    """Final per-row probabilities of the public rule for one native state per row."""
    P = np.exp(lsm(parent_lp))
    logP = np.log(np.maximum(P, 1e-15))
    return rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * native_lp), prior, sinkhorn, target_mix))


def test_like_groups(prompts, max_size, seed):
    """Hidden-test-like grouping: each prompt's answers split at random into groups of at most max_size."""
    prompts = np.asarray(prompts)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(prompts), np.int64)
    nxt = 0
    for prompt in np.unique(prompts):
        rows = rng.permutation(np.flatnonzero(prompts == prompt))
        for start in range(0, len(rows), max_size):
            out[rows[start:start + max_size]] = nxt
            nxt += 1
    return out


def native_rows(calibrate, plain, centered, row_fold, biases, states, groups, use_centered=True):
    """Calibrated native log-probs with each row's own fold state (public native_q8.calibrate, unchanged).
    Prompts never span folds, so per-fold calibration sees whole groups."""
    out = np.zeros(plain.shape, np.float64)
    for k in np.unique(row_fold):
        rows = np.flatnonzero(row_fold == k)
        out[rows] = calibrate(plain[rows][None], centered[rows][None], np.asarray(biases[k])[None],
                              groups[rows], [states[k]], use_centered)[0]
    return out


def ba(y, prediction):
    return float(np.mean([(prediction[y == c] == c).mean() for c in np.unique(y)]))


def paired_report(y, prompts, base, candidate, draws=2000, seed=0):
    """Delta balanced accuracy (points) with a prompt-cluster bootstrap 95% interval."""
    y, prompts = np.asarray(y), np.asarray(prompts)
    b, c = base.argmax(1), candidate.argmax(1)
    groups = [np.flatnonzero(prompts == q) for q in np.unique(prompts)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        diffs.append(ba(y[rows], c[rows]) - ba(y[rows], b[rows]))
    lower, upper = np.quantile(diffs, [.025, .975])
    return dict(rows=int(len(y)), base_ba=100 * ba(y, b), ba=100 * ba(y, c), delta=100 * (ba(y, c) - ba(y, b)),
                ci95=[100 * float(lower), 100 * float(upper)],
                fixed=int(((c == y) & (b != y)).sum()), harmed=int(((c != y) & (b == y)).sum()))
