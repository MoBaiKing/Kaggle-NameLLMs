"""OOF-weighted, ensemble-aware cross-entropy for the native expert (no RL estimator).

The policy is the deployed geometric rule before Sinkhorn, (1 - s) * log P_parent + s * log p_native,
renormalized; v2 optimized an arithmetic mixture that the 0.684 notebook never uses.

Why not the enumerated GRPO of v2: with a one-hot reward its unclipped objective is (q_y - mu) / sigma, whose
logit gradient has magnitude ~ sqrt(q_y (1 - q_y)), i.e. it vanishes on the rows the ensemble gets badly
wrong (q_y -> 0). Cross-entropy keeps gradient (1 - q_y) there, and the OOF weights decide which rows matter.
"""
import math

import torch

NATIVE_SHARE = .40


def ensemble_log_probs(native_lp, parent_lp, share=NATIVE_SHARE):
    if native_lp.shape != parent_lp.shape:
        raise ValueError('Native and parent score shapes differ')
    return ((1 - share) * parent_lp.detach().to(native_lp) + share * native_lp).log_softmax(-1)


def objective(native_lp, reference_lp, parent_lp, labels, weights, weight_scale, smoothing=.05, beta=.05):
    """weights: class balance x OOF difficulty per text; weight_scale: their mean over ALL training rows, so a
    batch of easy rows contributes a small gradient instead of being renormalized to a full one."""
    if not 0 <= smoothing < 1 or beta < 0 or weight_scale <= 0:
        raise ValueError('Invalid smoothing, KL weight or weight scale')
    policy = ensemble_log_probs(native_lp, parent_lp)
    nll = -policy.gather(1, labels[:, None])[:, 0]
    ce = (1 - smoothing) * nll - smoothing * policy.mean(-1)
    w = weights.to(policy)
    ce_term = (w * ce).sum() / (len(labels) * weight_scale)
    reference = reference_lp.detach().to(native_lp)
    kl = (native_lp.exp() * (native_lp - reference)).sum(-1).mean()
    loss = ce_term + beta * kl
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite v3 objective')
    accuracy = (policy.argmax(-1) == labels).float().mean()
    return loss, (ce_term.detach(), kl.detach(), nll.detach().mean(), accuracy)


def lr_factor(step, total, warmup_fraction=.1):
    """Linear warm-up over the first 10% of optimizer steps, then cosine to zero at the last step."""
    if total < 1 or not 0 <= step < total:
        raise ValueError('Step outside the fixed schedule')
    warm = max(1, math.ceil(warmup_fraction * total))
    if step < warm:
        return (step + 1) / warm
    span = max(1, total - warm)
    return .5 * (1 + math.cos(math.pi * (step - warm) / span))
