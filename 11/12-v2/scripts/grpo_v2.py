"""Full-support, old-policy-weighted clipped optimization for 18 author actions.

This is an enumerated categorical adaptation of GRPO, not finite sampled-group
GRPO. Exact reward moments and a clipped expectation replace Monte Carlo groups.
"""
import torch
import torch.nn.functional as F

NATIVE_SHARE = .40


def ensemble_log_probs(native, parent):
    """The frozen public OOF proxy: 60% parent log score + 40% native log score."""
    if native.shape != parent.shape:
        raise ValueError('Native and public parent score shapes differ')
    frozen = parent.detach().to(native)
    return torch.logaddexp(frozen + torch.log(native.new_tensor(1 - NATIVE_SHARE)),
                          native + torch.log(native.new_tensor(NATIVE_SHARE))).log_softmax(-1)


@torch.no_grad()
def enumerate_group(old_log_probs, labels):
    old = old_log_probs.detach().float().cpu()
    if old.ndim != 2 or len(old) != len(labels) or old.shape[-1] < 2:
        raise ValueError('Expected one categorical policy per labeled text')
    if not torch.isfinite(old).all() or not torch.allclose(old.exp().sum(-1), torch.ones(len(old)), atol=1e-5):
        raise ValueError('Old categorical policy is invalid')
    rewards = F.one_hot(labels.detach().cpu(), old.shape[-1]).float()
    probs = old.exp()
    mean = (probs * rewards).sum(-1, keepdim=True)
    variance = (probs * (rewards - mean).square()).sum(-1, keepdim=True)
    advantages = (rewards - mean) / variance.sqrt().clamp_min(1e-4)
    return dict(old_probs=probs, advantages=advantages,
                expected_reward=mean[:, 0], degenerate=(variance[:, 0] == 0).float())


def objective(policy_log_probs, native_log_probs, reference_native_log_probs, labels,
              sample_weights, group, epsilon=.2, beta=.02, ce_weight=.1):
    if not 0 < epsilon < 1 or beta < 0 or ce_weight < 0:
        raise ValueError('Invalid clipping, native KL or CE coefficient')
    old = group['old_probs'].to(policy_log_probs).detach()
    advantage = group['advantages'].to(policy_log_probs).detach()
    current = policy_log_probs.exp()
    # Algebraically old * min(ratio*A, clip(ratio)*A), with no division by tiny
    # old probabilities. Every action participates with its OLD policy mass.
    clipped_mass = torch.maximum(torch.minimum(current, (1 + epsilon) * old), (1 - epsilon) * old)
    surrogate = torch.minimum(current * advantage, clipped_mass * advantage).sum(-1)
    reference = reference_native_log_probs.to(native_log_probs).detach()
    kl = (native_log_probs.exp() * (native_log_probs - reference)).sum(-1)
    ce = F.nll_loss(policy_log_probs, labels, reduction='none')
    weights = sample_weights.to(policy_log_probs)
    average = lambda value: (weights * value).sum() / weights.sum().clamp_min(1e-6)
    pg, native_kl, anchor = -average(surrogate), average(kl), average(ce)
    loss = pg + beta * native_kl + ce_weight * anchor
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite enumerated categorical objective')
    clipped = (current < (1 - epsilon) * old) | (current > (1 + epsilon) * old)
    clip_mass = average((old * clipped).sum(-1))
    reward = average(current.gather(1, labels[:, None])[:, 0])
    return loss, (pg, native_kl, anchor, group['degenerate'].to(policy_log_probs).mean(), clip_mass, reward)
