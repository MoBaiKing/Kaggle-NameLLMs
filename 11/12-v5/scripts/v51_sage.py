"""Discriminative evidence replaces A2 in SAGE. Same numpy features for validation and deployment.
The gate is the expert route; no expert evidence => retain the incoming native/stack decision.
"""
import numpy as np
from v51_common import CLASSES, GROUPS
from v51_expert import route


def design(probs, expert_lp, enabled, sage):
    cand = sage.candidates(probs)
    routed = route(probs, enabled)
    X, _ = sage.features(probs, np.zeros_like(probs), cand, np.zeros((18, 18), bool))
    # Keep the existing conditional-logit model and relative final/rank features.
    # Disable typography (already present in the stack); add relative discriminative log probabilities.
    evidence = np.zeros(cand.shape); available = np.zeros(cand.shape)
    for j, group in enumerate(GROUPS):
        rows = np.flatnonzero(routed == j)
        if not len(rows):
            continue
        labels = np.array([CLASSES.index(c) for c in GROUPS[group]])
        block = expert_lp[np.ix_(rows, labels)]
        if not np.isfinite(block).all():
            raise ValueError(f'Missing expert evidence on routed rows: {group}')
        scores = expert_lp[rows[:, None], cand[rows]]
        mask = np.isin(cand[rows], labels)
        evidence[rows] = np.where(mask, np.clip(np.nan_to_num(scores) - block.max(1)[:, None], -20, 0), 0.)
        available[rows] = mask
    return np.concatenate([X[:, :, :4], evidence[..., None], available[..., None]], axis=-1), cand, routed >= 0


def fit(probs, expert_lp, enabled, y, sage):
    X, cand, gate = design(probs, expert_lp, enabled, sage)
    use, target = sage.training_target(cand, gate, y)
    if not use.any():
        return np.array([1., 0., 0., 0., 0., 0.])
    return sage.fit_logit(X[use], target[use], .3)  # Frozen before evaluation; no proxy weight or C search.


def apply(probs, expert_lp, enabled, weights, sage):
    X, cand, gate = design(probs, expert_lp, enabled, sage)
    return sage.decide(X, cand, gate, np.asarray(weights))
