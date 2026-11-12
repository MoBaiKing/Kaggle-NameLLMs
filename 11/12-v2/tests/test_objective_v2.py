"""Exact categorical objective, rare rewards, soft votes and detached teachers."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from grpo_v2 import enumerate_group, ensemble_log_probs, objective
from runtime_vote import multi_model_parent


def exact_gradient():
    logits = torch.tensor([[1., -1., .5], [-2., .4, 1.]], requires_grad=True)
    old = logits.detach().log_softmax(-1)
    labels = torch.tensor([1, 0])
    group = enumerate_group(old, labels)
    assert torch.allclose((group['old_probs'] * group['advantages']).sum(-1), torch.zeros(2), atol=1e-6)
    current = logits.log_softmax(-1)
    loss, _ = objective(current, current, old, labels, torch.ones(2), group, beta=0., ce_weight=0.)
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_loss = -(current.exp() * group['advantages']).sum(-1).mean()
    expected = torch.autograd.grad(expected_loss, logits)[0]
    assert torch.allclose(actual, expected, atol=1e-7)
    # At this policy the true action has probability .001. Enumeration keeps a
    # learning signal although a size-16 sampled group misses it ~98.4% of time.
    rare = torch.tensor([[.999, .001]]).log().requires_grad_()
    lp = rare.log_softmax(-1); target = torch.tensor([1])
    group = enumerate_group(lp, target)
    loss, _ = objective(lp, lp, lp.detach(), target, torch.ones(1), group, beta=0., ce_weight=0.)
    loss.backward()
    assert rare.grad[0, 1] < -.01 and group['degenerate'].item() == 0.


def clipping_and_sampling_limit():
    old = torch.tensor([[.6, .3, .1]]).log()
    current = torch.tensor([[.7, .1, .2]], requires_grad=True)
    labels = torch.tensor([2]); group = enumerate_group(old, labels)
    loss, parts = objective(current.log(), current.log(), old, labels, torch.ones(1), group,
                            beta=0., ce_weight=0.)
    ratio = current / old.exp()
    reference = -(old.exp() * torch.minimum(ratio * group['advantages'],
                   ratio.clamp(.8, 1.2) * group['advantages'])).sum()
    assert torch.allclose(loss, reference, atol=1e-7)
    actions = torch.multinomial(old.exp(), 200000, replacement=True,
                               generator=torch.Generator().manual_seed(8))
    terms = torch.minimum(ratio * group['advantages'], ratio.clamp(.8, 1.2) * group['advantages'])
    estimate = -terms.gather(1, actions).mean()
    assert abs(estimate.item() - loss.item()) < .004
    assert 0 < parts[4] <= 1.
    # Zero old support does not cause division-by-zero or NaN gradients.
    extreme = torch.tensor([[0., -1000., -1000.]], requires_grad=True)
    lp = extreme.log_softmax(-1); target = torch.tensor([2])
    group = enumerate_group(lp, target)
    loss, parts = objective(lp, lp, lp.detach(), target, torch.ones(1), group)
    loss.backward()
    assert torch.isfinite(extreme.grad).all() and torch.isfinite(torch.stack(parts)).all()


def voting():
    native = torch.tensor([[.2, .3, .5]], requires_grad=True)
    parent = torch.tensor([[.6, .3, .1]], requires_grad=True)
    blended = ensemble_log_probs(native.log(), parent.log())
    assert torch.allclose(blended.exp(), .6 * parent + .4 * native)
    blended[0, 2].backward()
    assert parent.grad is None and native.grad is not None
    terms = {'q14v5': np.log([[.6, .3, .1]]) * 3,
             'q8v5': np.log([[.2, .3, .5]]) * 2,
             'll8v5': np.log([[.1, .6, .3]])}
    public = np.array([[.4, .4, .2]])
    result = multi_model_parent(public, terms)
    expected = .75 * public + .25 * (3 * np.array([[.6, .3, .1]]) +
               2 * np.array([[.2, .3, .5]]) + np.array([[.1, .6, .3]])) / 6
    assert np.allclose(result, expected)
    try:
        multi_model_parent(public, {k: v for k, v in terms.items() if k != 'q14v5'})
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing strong-model vote was silently omitted')


if __name__ == '__main__':
    torch.set_num_threads(1)
    exact_gradient(); clipping_and_sampling_limit(); voting()
    print(json.dumps(dict(exact_expectation_gradient=True, rare_reward_signal=True,
                          clipping_matches_importance_ratio=True, monte_carlo_limit=True,
                          zero_support_safe=True, multi_model_vote=True, frozen_parent=True)))
