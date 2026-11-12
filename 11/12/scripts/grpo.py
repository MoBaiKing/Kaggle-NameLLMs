"""Categorical GRPO for the public native classifier (one author action per text).

Groups are independent samples for ONE text, not different answers to a prompt.
The finite action space permits exact KL instead of a sampled KL estimator.
"""
import torch
import torch.nn.functional as F


def aggregate_windows(logits, owners, weights, rows):
    weights = weights.to(logits)
    total = logits.new_zeros((rows, logits.shape[-1])).index_add(
        0, owners, logits * weights[:, None])
    denom = logits.new_zeros(rows).index_add(0, owners, weights)
    if (denom <= 0).any():
        raise ValueError('Every text needs at least one positive-weight window')
    return total / denom[:, None]


def native_log_probs(plain, centered, bias, groups, calibration):
    """Differentiable equivalent of public native_q8.calibrate, including singletons."""
    _, inv, counts = torch.unique(groups, return_inverse=True, return_counts=True)
    means = centered.new_zeros((len(counts), centered.shape[1])).index_add(0, inv, centered)
    relative = centered - means[inv] / counts[inv, None] + bias
    a = (plain.float() * calibration['plain_scale']).log_softmax(-1)
    b = (relative.float() * calibration['centered_scale']).log_softmax(-1)
    eligible = (counts[inv] >= 2) & (counts[inv] <= 18)
    fraction = eligible[:, None].to(a) * calibration['selected_centered_weight']
    return ((1 - fraction) * a + fraction * b).log_softmax(-1)


@torch.no_grad()
def rollout(old_log_probs, labels, group_size=16, seed=42):
    if group_size < 2:
        raise ValueError('GRPO requires at least two actions per text')
    # Sampling on CPU with a dedicated generator is identical on every rank and
    # does not disturb dropout/checkpoint RNG. Never inject the correct action.
    old = old_log_probs.detach().float().cpu()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    actions = torch.multinomial(old.exp(), group_size, replacement=True, generator=gen)
    rewards = (actions == labels.detach().cpu()[:, None]).float()
    centered = rewards - rewards.mean(-1, keepdim=True)
    std = rewards.std(-1, keepdim=True, unbiased=False)
    advantages = centered / std.clamp_min(1e-6)
    return dict(actions=actions, old_selected=old.gather(1, actions),
                advantages=advantages, rewards=rewards,
                zero_std=(std[:, 0] == 0).float())


def objective(log_probs, reference_log_probs, labels, sample_weights, samples,
              epsilon=.2, beta=.02, ce_weight=.1):
    if not 0 < epsilon < 1 or beta < 0 or ce_weight < 0:
        raise ValueError('Invalid GRPO clipping, KL or CE coefficient')
    actions = samples['actions'].to(log_probs.device)
    old = samples['old_selected'].to(log_probs).detach()
    adv = samples['advantages'].to(log_probs).detach()
    ratio = (log_probs.gather(1, actions) - old).exp()
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    surrogate = torch.minimum(ratio * adv, clipped * adv).mean(-1)
    ref = reference_log_probs.to(log_probs).detach()
    kl = (log_probs.exp() * (log_probs - ref)).sum(-1)
    ce = F.nll_loss(log_probs, labels, reduction='none')
    weights = sample_weights.to(log_probs)
    average = lambda x: (x * weights).sum() / weights.sum().clamp_min(1e-6)
    pg, kld, anchor = -average(surrogate), average(kl), average(ce)
    loss = pg + beta * kld + ce_weight * anchor
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite categorical GRPO objective')
    return loss, (pg, kld, anchor, samples['zero_std'].to(log_probs).mean(),
                  ((ratio - 1).abs() > epsilon).float().mean())
