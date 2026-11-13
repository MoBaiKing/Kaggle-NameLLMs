"""Objective = deployed geometric rule; CE keeps gradient on hard rows; schedule and WiSE are exact."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import objective_v3 as O
import proxy_v3 as X
import wise as W


def test():
    torch.manual_seed(0)
    native = torch.randn(6, 18).double().log_softmax(-1)
    parent = torch.randn(6, 18).double().log_softmax(-1)
    # Same numbers as infer.py: rowsoftmax((1 - share) * logP + share * lp), in log space.
    expected = np.log(X.rowsoftmax(.6 * parent.numpy() + .4 * native.numpy()))
    assert np.allclose(O.ensemble_log_probs(native, parent).numpy(), expected)
    labels = torch.arange(6)
    # Zero KL and zero KL gradient at the reference; weights scale the CE term exactly.
    leaf = native.clone().requires_grad_()
    loss, (ce, kl, nll, acc) = O.objective(leaf.log_softmax(-1), native, parent, labels, torch.ones(6), 1., .0, 1.)
    assert abs(float(kl)) < 1e-12
    loss2, (ce2, *_) = O.objective(native, native, parent, labels, torch.full((6,), 2.), 1., .0, 0.)
    assert torch.allclose(ce2, 2 * ce)
    loss3, (ce3, *_) = O.objective(native, native, parent, labels, torch.full((6,), 2.), 2., .0, 0.)
    assert torch.allclose(ce3, ce)
    # Label smoothing equals the textbook form.
    _, (cs, *_) = O.objective(native, native, parent, labels, torch.ones(6), 1., .05, 0.)
    pol = O.ensemble_log_probs(native, parent)
    manual = (-(.95 * pol[torch.arange(6), labels] + .05 * pol.mean(-1))).mean()
    assert torch.allclose(cs, manual)
    # A hard row (true-label probability ~0) still receives gradient under CE; enumerated GRPO's
    # sqrt(q (1 - q)) shape gives almost none there.
    logits = torch.full((1, 18), 0.).double(); logits[0, 0] = 9.
    leaf = logits.clone().requires_grad_()
    lp = leaf.log_softmax(-1)
    O.objective(lp, lp.detach(), torch.zeros(1, 18).double().log_softmax(-1), torch.tensor([5]), torch.ones(1), 1., 0., 0.)[0].backward()
    q = float(O.ensemble_log_probs(lp.detach(), torch.zeros(1, 18).double().log_softmax(-1))[0, 5].exp())
    assert leaf.grad.abs().sum() > 10 * math.sqrt(q * (1 - q)) * .4
    # Fixed schedule: warm-up to 1, cosine to ~0 at the end, never outside [0, 1].
    total = 135
    f = [O.lr_factor(s, total) for s in range(total)]
    warm = math.ceil(.1 * total)
    assert abs(f[warm - 1] - 1) < 1e-12 and f[0] == 1 / warm and f[-1] < 1e-3 and all(0 <= v <= 1 for v in f)
    try:
        O.lr_factor(total, total)
    except ValueError:
        pass
    else:
        raise AssertionError('Step outside the schedule accepted')
    # WiSE: exact float64 interpolation in the unchanged key layout.
    with tempfile.TemporaryDirectory() as d:
        a = {'plain.weight': torch.randn(18, 8), 'x.lora_A.default.weight': torch.randn(4, 8)}
        b = {k: v + torch.randn_like(v) for k, v in a.items()}
        save_file(a, f'{d}/a.safetensors'); save_file(b, f'{d}/b.safetensors')
        W.interpolate(f'{d}/a.safetensors', f'{d}/b.safetensors', .5, f'{d}/m.safetensors')
        m = load_file(f'{d}/m.safetensors')
        assert set(m) == set(a) and all(torch.allclose(m[k], (a[k] + b[k]) / 2) for k in a)
        W.interpolate(f'{d}/a.safetensors', f'{d}/b.safetensors', 1., f'{d}/one.safetensors')
        assert all(torch.equal(load_file(f'{d}/one.safetensors')[k], b[k]) for k in a)
    print(json.dumps(dict(objective_matches_deployed_geometric_rule=True, hard_row_gradient=True,
                          kl_zero_at_reference=True, schedule=True, wise_exact=True)))


if __name__ == '__main__':
    test()
