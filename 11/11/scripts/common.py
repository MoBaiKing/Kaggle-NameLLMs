"""Shared CPU code for 11/11: data/folds, public LoRA OOF stack, public native OOF, calibration fitting with the public
native_q8.calibrate semantics, test-like groupings. Validation only (official train rows)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
NW = R / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'          # public weights + calibration
NQ = R / 'external/native_q8_meta/ntl-native-q8-multidepth-fivefold-0679'     # public OOF + splits
sys.path.insert(0, str(R / 'vendor'))
import native_q8                                                              # noqa: E402  (public code, unchanged)

tr = pd.read_csv(R / 'data/original/train.csv', dtype=str); classes = sorted(tr.model.unique()); y_all = np.searchsorted(classes, tr.model.values)
pid = tr.prompt_id.values; upid = np.array(sorted(set(pid))); pidx = np.searchsorted(upid, pid); NP_ = len(upid); N = len(tr)
FOLDS = {int(k): np.array(v) for k, v in json.load(open(R / 'data/folds.json')).items()}


def lsm(z):
    z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def ba(t, p):
    return float(np.mean([(p[t == c] == c).mean() for c in np.unique(t)]))


def nll(lp, t):
    return float(-lp[np.arange(len(t)), t].mean())


def groups_sized(prompts, lo, hi, seed):
    """every prompt's rows split at random into groups of size drawn from [lo, hi]."""
    rng = np.random.default_rng(seed); g = np.zeros(len(prompts), np.int64); nxt = 0
    for q in np.unique(prompts):
        r = rng.permutation(np.where(prompts == q)[0]); i = 0
        while i < len(r):
            s = int(rng.integers(lo, hi + 1)); g[r[i:i + s]] = nxt; nxt += 1; i += s
    return g


def state_lp(plain, cent, bias, state, groups):
    """one fold state's calibrated log-probs via the public native_q8.calibrate."""
    return native_q8.calibrate(plain[None], cent[None], bias[None], groups, [state], True)[0]


T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 97))
W_GRID = (0.0, 0.15, 0.35, 0.5)
CONTEXTS = (('full', None), ('singletons', (1, 1)), ('sparse1_2_s42', (1, 2)), ('sparse1_5_s42', (1, 5)), ('sparse3_8_s42', (3, 8)))


def fit_calibration(plain, cent, bias, y, prompts):
    """public calibration fields fitted on inner rows: plain/centred temperature by NLL, centred weight by mean BA over
    five grouping contexts (as in the public calibration files). Ties go to the smaller centred weight."""
    plain, cent, bias = (np.asarray(x, np.float64) for x in (plain, cent, bias))
    ps = float(min(T_GRID, key=lambda s: nll(lsm(plain * s), y)))
    _, inv, cnt = np.unique(prompts, return_inverse=True, return_counts=True)
    means = np.zeros((len(cnt), plain.shape[1])); np.add.at(means, inv, cent); means /= cnt[:, None]
    el = cnt[inv] >= 2; rel = cent - means[inv] + bias
    cs = float(min(T_GRID, key=lambda s: nll(lsm(rel[el] * s), y[el])))
    ctx = {name: (pd.factorize(prompts)[0] if spec is None else (np.arange(len(y)) if spec == (1, 1) else groups_sized(prompts, *spec, 42)))
           for name, spec in CONTEXTS}
    grid = {}
    for w in W_GRID:
        st = dict(plain_scale=ps, centered_scale=cs, selected_centered_weight=w, outer_labels_used=False, centered_head_trained=True)
        c = {name: ba(y, state_lp(plain, cent, bias, st, g).argmax(1)) for name, g in ctx.items()}
        grid[str(w)] = dict(contexts=c, mean=float(np.mean(list(c.values()))))
    best = max(W_GRID, key=lambda w: (round(grid[str(w)]['mean'], 12), -w))
    return dict(plain_scale=ps, centered_head_trained=True, centered_scale=cs, centered_calibration_rows=int(len(y)), outer_labels_used=False,
                selected_centered_weight=float(best), selected_inner_score=grid[str(best)]['mean'], grid=grid)


def public_state(k):
    return json.load(open(NW / f'calibration/fold{k}.json'))['models']['qwen8']


def public_bias(k):
    from safetensors import safe_open
    with safe_open(str(NW / f'fold{k}/trainables.safetensors'), 'np') as f:
        return f.get_tensor('centered.bias').astype(np.float64)


def public_native_oof():
    """author's clean OOF (raw plain/centred head outputs) and the fold of every row; their folds == ours."""
    z = np.load(NQ / 'oof_reference.npz', allow_pickle=False); fold = np.full(N, -1)
    for k in range(5):
        v = np.array(json.load(open(NQ / f'fold{k}/split.json'))['validation_rows']); assert set(v) == set(FOLDS[k]); fold[v] = k
    assert (fold >= 0).all() and json.load(open(NW / 'manifest.json'))['classes'] == classes
    return z['plain'].astype(np.float64), z['centered'].astype(np.float64), fold


def public_lora_stack():
    """public LoRA fold adapters (q14v5 3.0 / q8v5 2.0 / ll8v5 1.0), clean OOF under the fold rule verified in earlier
    (infer.py --sim; adapter index matched label-free); returns the log of the weighted log-prob mean."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.RandomState(0); p = upid.copy(); rng.shuffle(p); m_ = {q: i % 5 for i, q in enumerate(p)}; sim = np.array([m_[q] for q in upid])
    W = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0}; S = np.zeros((N, len(classes)))
    for name, w in W.items():
        zz = np.load(R / f'data/reference/{name}.npz', allow_pickle=False); labels = [str(x) for x in zz['labels']]
        lp = zz['lp'].astype(np.float64)[:, :, [labels.index(c) for c in classes]]
        conf = np.stack([np.bincount(pidx, lp[k].max(1), NP_) / np.bincount(pidx, minlength=NP_) for k in range(5)])
        C = np.array([[conf[k][sim == f].mean() for k in range(5)] for f in range(5)])
        r_, c_ = linear_sum_assignment(C); mp = np.zeros(5, int); mp[r_] = c_; rf = mp[sim][pidx]
        S += w * lsm(lp[rf, np.arange(N)])
    return lsm(S / sum(W.values()))
