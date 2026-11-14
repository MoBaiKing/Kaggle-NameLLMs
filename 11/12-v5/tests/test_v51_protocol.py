"""CPU synthetic protocol tests; no 8B model is loaded or GPU result claimed."""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import v51_common as C
import v51_eb as EB
import v51_expert as E
import v51_data as D
import v51_sage as B
from v51_evaluate import choose_components, nested_sage, expert_scores
from v51_train import excluded_sets
from run_v51 import jobs, parse_lease
from build_v51 import validate_decision


class Protocol(unittest.TestCase):
    def test_dataframe_ids_are_saved_without_pickle(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'rows.npz'; ids = pd.Series(['001', '002']).astype(str).values
            C.save_npz(p, ids=ids, groups=np.array(['p1', 'p2'], object))
            with np.load(p, allow_pickle=False) as z:
                self.assertEqual(z['ids'].tolist(), ['001', '002'])
                self.assertEqual(z['groups'].tolist(), ['p1', 'p2'])
            with self.assertRaises(ValueError):
                C.save_npz(p, bad=np.array([{'arbitrary': 'object'}], object))

    def test_sparse_groups_never_mix_topics(self):
        prompt = np.repeat(np.arange(20), 18)
        groups = C.sparse_groups(prompt, 42)
        self.assertTrue(np.array_equal(groups, C.sparse_groups(prompt, 42)))
        self.assertLessEqual(np.bincount(groups).max(), 2)
        self.assertTrue(all(len(set(prompt[groups == g])) == 1 for g in np.unique(groups)))

    def test_prompt_fold_leak_rejected(self):
        frame = pd.DataFrame({'prompt_id': np.repeat(np.arange(5), 2)})
        split = {str(k): [2 * k, 2 * k + 1] for k in range(5)}
        self.assertEqual(C.row_folds(frame, split).tolist(), np.repeat(np.arange(5), 2).tolist())
        split['0'], split['1'] = [0, 2], [1, 3]
        with self.assertRaisesRegex(ValueError, 'prompt'):
            C.row_folds(frame, split)

    def test_eb_recovers_singleton_topic_signal(self):
        rng = np.random.default_rng(3); n = 180
        groups = np.repeat(np.arange(n), 3); y = np.tile(np.arange(3), n)
        topic = np.repeat(rng.normal(size=(n, 6)), 3, axis=0)
        loading = rng.normal(size=(6, 3)) * 8
        centered = topic @ loading + np.eye(3)[y] * 3
        model = EB.fit(topic[:360], centered[:360], groups[:360], .00001)
        state = dict(plain_scale=1., centered_scale=1., selected_centered_weight=.5)
        plain = np.zeros((1, 180, 3)); pc = centered[360:][None]; bias = np.zeros((1, 3))
        baseline = EB.calibrate(plain, pc, bias, topic[360:], np.arange(180), [state])
        candidate = EB.calibrate(plain, pc, bias, topic[360:], np.arange(180), [state], [model], [1e6])
        self.assertLess((baseline[0].argmax(1) == y[360:]).mean(), .4)
        self.assertGreater((candidate[0].argmax(1) == y[360:]).mean(), .95)

    def test_shrinkage_endpoints(self):
        z = np.arange(12).reshape(4, 3).astype(float); topic = np.zeros((4, 2)); group = np.array([0, 0, 1, 1])
        m = dict(projection=np.eye(2), mean=np.zeros(2), beta=np.zeros((2, 3)), intercept=np.array([2., 3., 4.]))
        observed, inv, _ = EB.aggregate(z, group)
        np.testing.assert_allclose(EB.relative(z, topic, group, 0, m, 0), z - observed[inv])
        np.testing.assert_allclose(EB.relative(z, topic, group, 0, m, 1e15), z - m['intercept'], atol=1e-12)

    def test_eb_pack_roundtrip(self):
        rng = np.random.default_rng(1); x = rng.normal(size=(30, 4)); z = rng.normal(size=(30, 18))
        model = EB.fit(x, z, np.repeat(np.arange(10), 3), .1)
        models, taus = EB.unpack(EB.pack([model] * 5, [16] * 5))
        np.testing.assert_allclose(EB.predict(models[2], x), EB.predict(model, x))
        self.assertEqual(taus.tolist(), [16] * 5)

    def test_balancing_keeps_all_large_class_rows(self):
        labels = np.array([0] * 3 + [1] * 10 + [2] * 5)
        rows = E.balanced_order(labels, np.random.default_rng(42))
        self.assertEqual(set(rows), set(range(len(labels))))
        self.assertEqual(np.bincount(labels[rows]).tolist(), [10, 10, 10])

    def test_tiny_lora_freezes_base_and_preserves_initial_output(self):
        import torch
        from torch import nn
        encoder = nn.Module(); encoder.block = nn.Module(); encoder.block.q_proj = nn.Linear(5, 4)
        x = torch.randn(3, 5); expected = encoder.block.q_proj(x).detach()
        self.assertEqual(E.attach_lora(encoder), 1)
        encoder.eval(); actual = encoder.block.q_proj(x)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        layer = encoder.block.q_proj
        self.assertIsNone(layer.base_layer.weight.grad)
        self.assertGreater(float(layer.b.weight.grad.abs().sum()), 0.)
        self.assertEqual(set(E.state(encoder)), {'block.q_proj.a.weight', 'block.q_proj.b.weight'})

    def test_disjoint_one_expert_route(self):
        probs = np.ones((4, 18)) * .001
        for i, pair in enumerate([(6, 7), (0, 9), (11, 16), (2, 6)]):
            probs[i, pair[0]], probs[i, pair[1]] = .6, .3
        self.assertEqual(E.route(probs, list(C.GROUPS)).tolist(), [0, 1, 2, -1])
        self.assertEqual(E.route(probs, ['gemini']).tolist(), [0, -1, -1, -1])

    def test_fold_route_is_deterministic_and_covers_folds(self):
        folds = [E.deploy_fold(str(i)) for i in range(200)]
        self.assertEqual(set(folds), set(range(5)))
        self.assertEqual(folds, [E.deploy_fold(str(i)) for i in range(200)])

    def test_kaggle_fp16_expert_shards_like_native(self):
        mapping = E.inference_device_map()
        self.assertEqual(mapping['embed_tokens'], 0)
        self.assertEqual(mapping['layers.15'], 0)
        self.assertEqual(mapping['layers.16'], 1)
        self.assertEqual(mapping['layers.35'], 1)
        self.assertEqual(mapping['norm'], 1)

    def test_pair_exclusions_protect_outer_fold(self):
        exclusions = excluded_sets()
        self.assertEqual(len(exclusions), 15)
        for outer in range(5):
            for inner in range(5):
                if outer != inner:
                    self.assertIn(tuple(sorted((outer, inner))), exclusions)

    def test_expert_completion_and_excluded_rows_are_required(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run = root / 'gemini/exclude_0'; run.mkdir(parents=True)
            (run / 'final.safetensors').write_bytes(b'tiny protocol fixture')
            official = pd.DataFrame({'id': [str(i) for i in range(10)]}); fold = np.repeat(np.arange(5), 2)
            C.save_npz(run / 'scores.npz', rows=np.array([0, 1]), ids=official.id.iloc[:2].values,
                       lp=np.zeros((2, 3)), classes=np.array(C.GROUPS['gemini']))
            rec = dict(payload_sha256=C.sha(run / 'scores.npz'), checkpoint_sha256=C.sha(run / 'final.safetensors'),
                       excluded=[0], smoke=False, identity=dict(official_sha256=C.sha(C.BASE / 'data/original/train.csv'),
                       folds_sha256=C.sha(C.BASE / 'data/folds.json'), plan_sha256=C.sha(C.ROOT / 'artifacts/v51_plan.json'),
                       code_sha256=C.sha(C.ROOT / 'scripts/v51_train.py'), expert_sha256=C.sha(C.ROOT / 'scripts/v51_expert.py')))
            C.dump(run / 'scores.json', rec)
            with self.assertRaisesRegex(ValueError, 'completion'):
                expert_scores(root, 'gemini', (0,), official, fold, {})
            C.dump(run / 'complete.json', rec)
            self.assertEqual(expert_scores(root, 'gemini', (0,), official, fold, {}).shape, (10, 18))
            rec['excluded'] = [0, 1]; C.dump(run / 'scores.json', rec); C.dump(run / 'complete.json', rec)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                expert_scores(root, 'gemini', (0,), official, fold, {})

    def test_sage_training_consumes_double_exclusion_scores(self):
        fold = np.repeat(np.arange(5), 2); probs = np.zeros((10, 18)); probs[:, 0] = np.arange(10) + 1
        scores = {}
        for excluded in excluded_sets():
            out = np.full((10, 18), np.nan)
            rows = np.flatnonzero(np.isin(fold, excluded))
            out[np.ix_(rows, [6, 7, 8])] = 100 * sum(k + 1 for k in excluded) + rows[:, None]
            scores[excluded] = out
        calls = []
        def fit(p, expert, enabled, y, sage):
            outer = len(calls); rows = p[:, 0].astype(int) - 1
            self.assertFalse((fold[rows] == outer).any())
            for i, row in enumerate(rows):
                pair = tuple(sorted((outer, int(fold[row]))))
                np.testing.assert_array_equal(expert[i, 6:9], scores[pair][row, 6:9])
            calls.append(outer); return np.zeros(6)
        with patch.object(B, 'fit', side_effect=fit), patch.object(B, 'apply', side_effect=lambda p, *args: np.zeros(len(p), int)):
            nested_sage(probs, np.full(10, 6), fold, {'gemini': scores}, ['gemini'], None)
        self.assertEqual(calls, list(range(5)))

    def test_bootstrap_conditional_uses_changing_denominator(self):
        y = np.tile([6, 7, 9, 10], 30); base = y.copy(); new = y.copy()
        base[::4] = 7  # Family right, version wrong.
        new[1::12] = 9  # Cross-family errors alter the denominator.
        rec = C.paired_ci(y, np.repeat(np.arange(30), 4), base, new, 'conditional', draws=300)
        self.assertAlmostEqual(rec['delta'], 100 * (C.conditional(y, new) - C.conditional(y, base)))
        self.assertGreater(rec['ci95'][0], 0)

    def test_gate_all_fail(self):
        result = choose_components({'ci95': [-.2, .5]}, {})
        self.assertEqual(result, dict(eb=False, experts=[], build=False))

    def test_gate_partial_and_combined(self):
        groups = {'gemini': {'within_delta': 2., 'family': {'ci95': [.1, 3.]}},
                  'glm_claude': {'within_delta': -1., 'family': {'ci95': [.1, 3.]}}}
        result = choose_components({'ci95': [-.1, 1.]}, groups, {'ci95': [.1, 3.]})
        self.assertEqual(result['experts'], ['gemini'])
        self.assertFalse(result['eb'])
        self.assertEqual(choose_components({'ci95': [.01, .5]}, groups, {'ci95': [-.1, 3.]})['experts'], [])

    def test_stale_gate_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t); artifact = path / 'scores'; artifact.write_text('verified')
            d = dict(method='v51_eb_sibling_experts', enabled=dict(eb=True, experts=[], build=True),
                     eb={'ci95': [.1, .5]}, groups={}, combined=None, proxy_used_for_acceptance=False,
                     hidden_test_evaluated=False, artifacts={str(artifact): C.sha(artifact)})
            C.dump(path / 'decision.json', d); validate_decision(path / 'decision.json')
            artifact.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Stale'):
                validate_decision(path / 'decision.json')

    def test_no_manual_gate_override(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'decision.json'
            C.dump(path, dict(method='v51_eb_sibling_experts', enabled=dict(eb=True, experts=[], build=True),
                             eb={'ci95': [-.1, .5]}, groups={}))
            with self.assertRaises(ValueError):
                validate_decision(path)

    def test_normalized_duplicates_and_official_copies_removed(self):
        raw = pd.DataFrame(dict(id=['a', 'b', 'c', 'd'], text=[' official  x ', 'same', 'same', 'novel'],
                   model=[C.CLASSES[6], C.CLASSES[6], C.CLASSES[7], C.CLASSES[8]],
                   prompt_canon=['p'] * 4, source_dataset=['s'] * 4))
        out = D.clean(raw, pd.DataFrame({'text': ['official x']}))
        self.assertEqual(out.text.tolist(), ['novel'])

    def test_four_distinct_job_commands(self):
        command = jobs(False, Path('/tmp/fixture'), '2099-01-01T00:00:00+08:00')
        self.assertEqual([n for n, _ in command], [*C.GROUPS, 'eb'])
        self.assertTrue(all('torch.distributed.run' not in ' '.join(c) for _, c in command))

    def test_lease_rejects_missing_timezone_and_duplicate_gpu(self):
        for d in [dict(deadline='2026-11-15T14:00:00', gpus=[0, 1], allowed=True),
                  dict(deadline='2099-11-15T14:00:00+08:00', gpus=[0, 0], allowed=True)]:
            with self.assertRaises(ValueError):
                parse_lease(d)


if __name__ == '__main__':
    unittest.main()
