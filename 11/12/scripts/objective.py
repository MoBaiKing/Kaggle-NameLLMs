"""Same-prompt, different-author ranking; no inference-time assignment."""
import torch
import torch.nn.functional as F


def prompt_pair_loss(logits, labels, weights, groups, temperature=1.0):
    # For i,j answering the same prompt, both correct labels must beat swapping
    # their authors. Shared prompt-specific class offsets cancel in this margin.
    if temperature <= 0:
        raise ValueError('Pair temperature must be positive')
    i, j = torch.triu_indices(len(labels), len(labels), offset=1, device=labels.device)
    keep = (groups[i] == groups[j]) & (labels[i] != labels[j])
    i, j = i[keep], j[keep]
    if not len(i):
        return logits.sum() * 0
    margin = .5 * (logits[i, labels[i]] - logits[i, labels[j]]
                    + logits[j, labels[j]] - logits[j, labels[i]])
    w = (weights[i] * weights[j]).sqrt()
    return (F.softplus(-margin / temperature) * w).sum() / w.sum().clamp_min(1e-6)
