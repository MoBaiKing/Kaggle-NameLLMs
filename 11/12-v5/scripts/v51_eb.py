"""Empirical Bayes centering, numpy only; this same file runs in Kaggle.
Predict the group mean from frozen Qwen topic vectors, then shrink the observed mean toward it.
No extra transformer forward; no logit/weight rewrite. Public temperatures/mixture weight remain frozen.
"""
import numpy as np
from v51_common import lsm


def projection(width, rank=128):
    return (np.random.default_rng(42).normal(size=(width, min(width, rank))) / np.sqrt(width)).astype(np.float32)


def aggregate(x, groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), x.shape[1]), np.float64)
    np.add.at(sums, inverse, x)
    return sums / counts[:, None], inverse, counts


def fit(topic, centered, groups, alpha):
    """Equal weight per prompt. Only external data enters the ridge regression."""
    P = projection(topic.shape[1]); x, _, _ = aggregate(topic @ P, groups)
    z, _, _ = aggregate(centered, groups)
    xm, zm = x.mean(0), z.mean(0); xc, zc = x - xm, z - zm
    beta = np.linalg.solve(xc.T @ xc + alpha * len(x) * np.eye(x.shape[1]), xc.T @ zc)
    return dict(projection=P, mean=xm, intercept=zm, beta=beta)


def predict(model, topic):
    return (topic @ model['projection'] - model['mean']) @ model['beta'] + model['intercept']


def relative(centered, topic, groups, bias, model, tau):
    observed, inverse, counts = aggregate(centered, groups)
    predicted, _, _ = aggregate(predict(model, topic), groups)
    weight = counts / (counts + tau)
    mean = weight[:, None] * observed + (1 - weight[:, None]) * predicted
    return centered - mean[inverse] + bias


def calibrate(plain, centered, bias, topic, groups, states, models=None, taus=None):
    out = []
    for k, state in enumerate(states):
        if models is None:
            means, inverse, counts = aggregate(centered[k], groups)
            rel = centered[k] - means[inverse] + bias[k]
            weight = np.where((counts[inverse] >= 2) & (counts[inverse] <= 18),
                              state['selected_centered_weight'], 0.)[:, None]
        else:
            rel = relative(centered[k], topic, groups, bias[k], models[k], taus[k])
            weight = state['selected_centered_weight']
        a = lsm(plain[k] * state['plain_scale']); b = lsm(rel * state['centered_scale'])
        out.append(lsm((1 - weight) * a + weight * b))
    return np.asarray(out)


def select_external(topic, centered, groups):
    """Prompt-nested external calibration. No official labels select alpha or tau.
    Noise/prior ratio is chosen using sparse observed groups against their full external prompt mean.
    """
    P = projection(topic.shape[1]); reduced = topic @ P
    unique = np.unique(groups)
    split = {g: i % 5 for i, g in enumerate(np.random.default_rng(42).permutation(unique))}
    fold = np.array([split[g] for g in groups]); errors = {}
    for alpha in (.001, .01, .1, 1.):
        for tau in (4., 16., 64.):
            errors[(alpha, tau)] = 0.
        count = 0
        for k in range(5):
            tr, te = fold != k, fold == k
            x, _, _ = aggregate(reduced[tr], groups[tr]); z, _, _ = aggregate(centered[tr], groups[tr])
            xm, zm = x.mean(0), z.mean(0); xc = x - xm
            beta = np.linalg.solve(xc.T @ xc + alpha * len(x) * np.eye(x.shape[1]), xc.T @ (z - zm))
            true, inv, _ = aggregate(centered[te], groups[te])
            pred = (reduced[te] - xm) @ beta + zm; obs = centered[te]
            for tau in (4., 16., 64.):
                estimate = obs / (1 + tau) + tau * pred / (1 + tau)
                errors[(alpha, tau)] += float(((estimate - true[inv]) ** 2).sum())
            count += obs.size
        for tau in (4., 16., 64.):
            errors[(alpha, tau)] /= count
    alpha, tau = min(errors, key=errors.get)
    return fit(topic, centered, groups, alpha), tau, dict(alpha=alpha, tau=tau, mse=errors[(alpha, tau)],
                                                        selection='external_prompt_folds_only')


def pack(models, taus):
    out = {'taus': np.asarray(taus)}
    for k, model in enumerate(models):
        out.update({f'{k}_{name}': value for name, value in model.items()})
    return out


def unpack(arrays):
    return [{n: arrays[f'{k}_{n}'] for n in ('projection', 'mean', 'intercept', 'beta')}
            for k in range(5)], arrays['taus']
