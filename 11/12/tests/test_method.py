"""Pair-margin invariance and exact, reproducible native-window sampling."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from objective import prompt_pair_loss
from windows import INSTRUCTION, LIMIT, TERMINAL, TokenStore, write_cache
from shared import BASE
from selection import assess, choose


class Tokens:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        def encode(t):
            return [ord(c) + 1 for c in t]
        if text == '' and add_special_tokens:
            return {'input_ids': [TERMINAL]}
        return {'input_ids': encode(text) if isinstance(text, str) else [encode(t) for t in text]}


def pairs():
    x = torch.tensor([[3., -1., 2.], [-2., 4., 1.], [.5, 2., 3.], [1., 2., .3]], requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0]); weights = torch.tensor([.5, 2., 1., 1.5]); groups = torch.tensor([0, 0, 1, 1])
    loss = prompt_pair_loss(x, labels, weights, groups)
    shifts = torch.tensor([[100., -10., 7.], [-3., 8., 14.]])
    assert torch.allclose(loss, prompt_pair_loss(x + shifts[groups], labels, weights, groups), atol=1e-6)
    assert prompt_pair_loss(x.flip(0), labels.flip(0), weights.flip(0), groups.flip(0)).allclose(loss)
    strong = torch.tensor([[5., -5.], [-5., 5.]])
    weak = -strong
    args = (torch.tensor([0, 1]), torch.ones(2), torch.zeros(2, dtype=torch.long))
    assert prompt_pair_loss(strong, *args) < prompt_pair_loss(weak, *args)
    loss.backward(); assert torch.isfinite(x.grad).all()
    empty = prompt_pair_loss(x, labels, weights, torch.arange(4)); assert empty.item() == 0
    empty.backward(); assert torch.isfinite(x.grad).all()
    print(json.dumps(dict(prompt_offset_invariance=True, pair_ranking_gradient=True)))


def windows():
    rng = np.random.default_rng(4)
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
    texts = ['short'] + [''.join(rng.choice(alphabet, n)) for n in (900, 1900)]
    tokenizer = Tokens()
    with tempfile.TemporaryDirectory(prefix='native_windows_') as temp:
        target = Path(temp) / 'cache'
        store = write_cache(texts, tokenizer, target, {'fixture': True})
        prefix = tokenizer(INSTRUCTION)['input_ids']
        all_sequences = []
        for row, text in enumerate(texts):
            seqs, weights = store.all_windows(row)
            tokens = tokenizer(INSTRUCTION + text)['input_ids']
            expected = [tokens[:511] + [TERMINAL]]
            expected += [prefix + tokens[p:p + 511 - len(prefix)] + [TERMINAL]
                         for p in range(511, len(tokens), 511 - len(prefix))]
            assert seqs == expected and all(len(seq) <= LIMIT for seq in seqs)
            assert store.select(row, first=True) == seqs[0]
            all_sequences.extend(seqs)
            draws = [store.select(row, step=step) for step in range(5000)]
            observed = np.array([sum(value == seq for value in draws) for seq in seqs]) / len(draws)
            assert np.allclose(observed, weights / weights.sum(), atol=.03), (observed, weights)
            reopened = TokenStore(target)
            for step in (0, 23, 100, 4999):
                assert store.select(row, step=step) == reopened.select(row, step=step)
        if (BASE / 'vendor/native_q8.py').exists():
            sys.path.insert(0, str(BASE / 'vendor'))
            from native_q8 import windows as native_windows
            seqs, owners, weights = native_windows(tokenizer, texts)
            assert seqs == all_sequences
            assert weights.tolist() == [weight for row in range(3) for weight in store.all_windows(row)[1]]
        # Corrupting the cached token bytes must be detected before reuse.
        with (target / 'tokens.bin').open('r+b') as out:
            out.write(b'bad!')
        try:
            TokenStore(target)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed cache was accepted')
        print(json.dumps(dict(native_window_parity=True, weighted_sampling=True,
                              deterministic_resume=True, cache_integrity=True)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    pairs(); windows()
    y = np.tile(np.arange(3), 20); groups = np.repeat(np.arange(20), 3)
    good = np.eye(3)[y]
    bad = np.eye(3)[(y + 1) % 3]
    assert assess(y, groups, bad, good, .8)['passed']
    assert not assess(y, groups, good, bad, .8)['passed']
    assert not assess(y, groups, good, good, .8)['passed']
    assert choose({'ours4': {'passed': True}, 'ours2': {'passed': True}}) == ('ours4', [0, 1, 2, 3])
    try:
        choose({})
    except ValueError:
        pass
    else:
        raise AssertionError('Missing optimized candidates must stop the build')
    print(json.dumps(dict(fixed_candidate_gate=True, degraded_model_rejected=True)))
