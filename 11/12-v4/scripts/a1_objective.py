"""A1 (HSC-Native) auxiliary losses on top of the v3 objective.

hierarchical_supcon: supervised contrastive loss on the native projection with graded positives, after the
multi-level idea of DeTeCtive (NeurIPS 2024): same model weight 1, same family / fixed confusable pair weight
FAMILY_POS, everything else negative. sibling_margin: the true label must beat the log-sum-exp of its siblings
and fixed confusables by MARGIN nats (the 61% of public-rule errors that stay inside a family).
"""
import numpy as np
import torch
import torch.nn.functional as F

import families as FAM

NEIGHBORS = torch.tensor(FAM.NEIGHBORS)
FAMILY_ID = torch.tensor(FAM.FAMILY_ID)


def ensemble_objective(native_lp, reference_lp, parent_lp, labels, weights, weight_scale, share, smoothing=.05, beta=.05):
    """v3 objective (objective_v3.objective) with the native share as an argument: v4 trains against the
    LB-anchored proxy policy log_softmax((1 - share) log P_parent + share log p_native), share = proxy_v4.ANCHOR_SHARE."""
    policy = ((1 - share) * parent_lp.detach().to(native_lp) + share * native_lp).log_softmax(-1)
    nll = -policy.gather(1, labels[:, None])[:, 0]
    ce = (1 - smoothing) * nll - smoothing * policy.mean(-1)
    ce_term = (weights.to(policy) * ce).sum() / (len(labels) * weight_scale)
    kl = (native_lp.exp() * (native_lp - reference_lp.detach().to(native_lp))).sum(-1).mean()
    loss = ce_term + beta * kl
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite A1 objective')
    return loss, (ce_term.detach(), kl.detach(), nll.detach().mean(), (policy.argmax(-1) == labels).float().mean())


def hierarchical_supcon(z, labels, temperature=.1, family_pos=.25):
    """z: [n, d] L2-normalized. Rows without any positive are skipped. Returns a scalar."""
    n = len(labels)
    if n < 2:
        return z.sum() * 0
    neighbors, fam = NEIGHBORS.to(labels.device), FAMILY_ID.to(labels.device)
    same = labels[:, None] == labels[None, :]
    related = (fam[labels][:, None] == fam[labels][None, :]) | neighbors[labels][:, labels]
    weight = same.float() + family_pos * (related & ~same).float()
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    weight = weight.masked_fill(eye, 0)
    logits = (z.float() @ z.float().T / temperature).masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    total = weight.sum(1)
    keep = total > 0
    if not keep.any():
        return z.sum() * 0
    return (-(weight * log_prob).sum(1)[keep] / total[keep]).mean()


def sibling_margin(log_probs, labels, margin=1.0):
    """softplus(margin - (lp_y - logsumexp_{s in N(y)} lp_s)) on rows whose label has neighbors."""
    neighbors = NEIGHBORS.to(labels.device)[labels]
    has = neighbors.any(1)
    if not has.any():
        return log_probs.sum() * 0
    true = log_probs.gather(1, labels[:, None])[:, 0]
    rival = torch.logsumexp(log_probs.masked_fill(~neighbors, -1e4), 1)
    return F.softplus(margin - (true - rival))[has].mean()


def pool_ce(native_lp, labels, smoothing=.05):
    return ((1 - smoothing) * -native_lp.gather(1, labels[:, None])[:, 0] - smoothing * native_lp.mean(-1)).mean()


def pool_cells(labels, prompts, rows, n_cells, size, rng, p_global=.25):
    """Sibling-structured replay cells: each cell draws one confusion group (a family with >= 2 models or a fixed
    cross pair) or, with probability p_global, any classes; rows are spread evenly over the group's classes and
    taken from distinct prompt clusters where possible. Deterministic for a given rng."""
    groups = [np.flatnonzero(FAM.FAMILY == f) for f in np.unique(FAM.FAMILY) if (FAM.FAMILY == f).sum() >= 2]
    index = {c: i for i, c in enumerate(FAM.CLASSES)}
    groups += [np.array([index[a], index[b]]) for a, b in FAM.CROSS_PAIRS]
    by_class = {c: list(rng.permutation(rows[labels[rows] == c])) for c in range(len(FAM.CLASSES))}
    cells = []
    for _ in range(n_cells):
        if rng.uniform() < p_global:
            classes = rng.permutation(len(FAM.CLASSES))[:size]
        else:
            classes = groups[rng.integers(len(groups))]
        cell, seen, i = [], set(), 0
        while len(cell) < size and i < 8 * size:
            c = int(classes[i % len(classes)]); i += 1
            pool = by_class[c]
            for _ in range(len(pool)):
                r = pool.pop()
                pool.insert(0, r)
                if prompts[r] not in seen:
                    cell.append(int(r)); seen.add(prompts[r])
                    break
        cells.append(cell)
    return cells

