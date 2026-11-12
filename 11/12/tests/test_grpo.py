"""CPU mathematical checks, including sparse rewards and native policy semantics."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from grpo import aggregate_windows, native_log_probs, objective, rollout


CAL = dict(plain_scale=1.1, centered_scale=.8, selected_centered_weight=.35)


def policy_check():
    torch.manual_seed(7)
    p, c, b = torch.randn(22, 3), torch.randn(22, 3), torch.randn(3)
    groups = torch.tensor([0, 1, 1] + [2] * 19)
    actual = native_log_probs(p, c, b, groups, CAL).numpy()
    lsm = lambda x: x - np.logaddexp.reduce(x, axis=-1, keepdims=True)
    relative = c.numpy().copy()
    for group in np.unique(groups):
        ix = groups.numpy() == group
        relative[ix] -= relative[ix].mean(0)
    relative += b.numpy()
    a, d = lsm(p.numpy() * CAL['plain_scale']), lsm(relative * CAL['centered_scale'])
    fractions = np.array([0., .35, .35] + [0.] * 19)[:, None]
    expected = lsm((1 - fractions) * a + fractions * d)
    assert np.allclose(actual, expected, atol=1e-6)
    x = torch.randn(4, 3, requires_grad=True)
    owners, weights = torch.tensor([0, 0, 1, 1]), torch.tensor([1., 3., 2., 2.])
    avg = aggregate_windows(x, owners, weights, 2)
    assert torch.allclose(avg[0], (x[0] + 3 * x[1]) / 4)
    avg.sum().backward()
    assert torch.allclose(x.grad[:, 0], torch.tensor([.25, .75, .5, .5]))


def reward_check():
    old = torch.zeros(2, 3).log_softmax(-1)
    labels = torch.tensor([0, 2])
    samples = rollout(old, labels, 128, 4)
    assert torch.allclose(samples['advantages'].mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.equal(samples['rewards'], (samples['actions'] == labels[:, None]).float())
    assert all(torch.equal(v, rollout(old, labels, 128, 4)[k]) for k, v in samples.items())
    logits = torch.zeros(1, 3, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=.5)
    initial = logits.softmax(-1)[0, 0].item()
    for step in range(12):
        lp = logits.log_softmax(-1)
        draw = rollout(lp, labels[:1], 256, step)
        loss, _ = objective(lp, old[:1], labels[:1], torch.ones(1), draw, beta=0., ce_weight=0.)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert logits.softmax(-1)[0, 0] > initial + .3
    # No success in a group gives zero policy gradient, rather than NaN or a
    # fabricated positive action. CE remains an explicitly separate objective.
    certain = torch.tensor([[0., -1000., -1000.]], requires_grad=True)
    lp = certain.log_softmax(-1)
    draw = rollout(lp, torch.tensor([2]), 16)
    loss, parts = objective(lp, lp.detach(), torch.tensor([2]), torch.ones(1), draw, beta=0., ce_weight=0.)
    loss.backward()
    assert draw['zero_std'].item() == 1 and torch.count_nonzero(certain.grad) == 0
    assert torch.isfinite(torch.stack(parts)).all()


def clipping_check():
    old = torch.tensor([[.5, .5]]).log()
    current = torch.tensor([[.9, .1]], requires_grad=True)
    samples = dict(actions=torch.tensor([[0, 1]]), old_selected=old,
                   advantages=torch.tensor([[1., -1.]]), zero_std=torch.zeros(1))
    loss, parts = objective(current.log(), old, torch.tensor([0]), torch.ones(1), samples,
                            epsilon=.2, beta=0., ce_weight=0.)
    assert torch.allclose(loss, torch.tensor(-.2), atol=1e-6)
    loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))
    assert parts[-1].item() == 1.
    x = torch.randn(3, 4, requires_grad=True)
    lp = x.log_softmax(-1)
    samples = rollout(lp, torch.tensor([0, 1, 2]), 16)
    _, parts = objective(lp, lp.detach(), torch.tensor([0, 1, 2]), torch.ones(3), samples)
    assert abs(parts[1].item()) < 1e-7
    parts[1].backward()
    assert x.grad.abs().max() < 1e-7


if __name__ == '__main__':
    torch.set_num_threads(1)
    policy_check(); reward_check(); clipping_check()
    print(json.dumps(dict(native_policy_parity=True, window_gradient=True,
                          correct_action_reward=True, zero_variance_safe=True,
                          clipped_surrogate=True, exact_kl=True)))
