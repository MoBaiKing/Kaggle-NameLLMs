"""CPU checks against the pinned public notebook, including all three accepted-component combinations."""
import json
import types
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_v51 as B
from v51_common import dump, sha, ROOT


class Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = B.SRC / 'ntl-684-ozertuu-mirror.ipynb'
        if not cls.original.exists():
            raise RuntimeError('Prepare pinned Notebook: python ../11/scripts/assets.py notebook')
        if sha(cls.original) != B.PIN:
            raise ValueError('Pinned Notebook differs')
        cls.nb = json.loads(cls.original.read_text())
        cls.engine = next(''.join(c['source']) for c in cls.nb['cells'] if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n'))

    def test_all_component_patches_compile(self):
        for eb, experts in [(True, []), (False, ['gemini']), (True, ['gemini'])]:
            text = B.edit_infer(self.engine, dict(eb=eb, experts=experts, build=True))
            compile(text.split('\n', 1)[1], 'infer.py', 'exec')
            self.assertEqual('v51_runtime.native(' in text, eb)
            self.assertEqual('v51_runtime.rerank(' in text, bool(experts))
            self.assertIn("folds == list(range(5))", text)
            self.assertNotIn("log('NATIVE SKIPPED", text)
            self.assertEqual('        del E_q' in text, not eb)

    def test_changed_anchor_fails(self):
        with self.assertRaisesRegex(ValueError, 'anchor'):
            B.edit_infer(self.engine.replace(B.CALIBRATE, 'changed'), dict(eb=True, experts=[], build=True))

    def test_public_calibration_matches_pinned_code(self):
        import v51_eb as E
        source = next(''.join(c['source']).partition('\n')[2] for c in self.nb['cells']
                      if ''.join(c.get('source', [])).startswith('%%writefile native_q8.py\n'))
        module = types.ModuleType('pinned_native_test'); exec(compile(source, 'native_q8.py', 'exec'), module.__dict__)
        rng = np.random.default_rng(8); pp = rng.normal(size=(5, 36, 18)); pc = rng.normal(size=pp.shape)
        bias = rng.normal(size=(5, 18)); groups = np.repeat(np.arange(12), 3); groups[:3] = np.arange(3) + 99
        states = [dict(plain_scale=.7, centered_scale=1.3, selected_centered_weight=.35,
                       outer_labels_used=False, centered_head_trained=True)] * 5
        actual = E.calibrate(pp, pc, bias, np.zeros((36, 5)), groups, states)
        np.testing.assert_array_equal(actual, module.calibrate(pp, pc, bias, groups, states, True))

    def test_build_eb_only_is_minimal_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); (folder / 'eb_models.npz').write_bytes(b'synthetic_test_payload')
            artifact = folder / 'eb_models.npz'
            decision = dict(method='v51_eb_sibling_experts', enabled=dict(eb=True, experts=[], build=True),
                            eb=dict(ci95=[.1, 1.]), groups={}, combined=None, hidden_test_evaluated=False,
                            proxy_used_for_acceptance=False, artifacts={str(artifact): sha(artifact)})
            dump(folder / 'decision.json', decision)
            built = B.build(folder / 'decision.json', folder / 'bundle', 'testuser')
            spec = json.loads((folder / 'bundle/upload' / B.SPEC).read_text())
            self.assertTrue(spec['enabled']['eb']); self.assertFalse(spec['enabled']['experts'])
            self.assertFalse(any(n.endswith('.safetensors') for n in spec['sha256']))
            nb = json.loads((folder / 'bundle/final/submission.ipynb').read_text())
            # Original cells remain in order: exactly setup and infer changed, generated files inserted once.
            original_non_engine = [''.join(c['source']) for i, c in enumerate(self.nb['cells'])
                                   if i != 2 and not ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
            new_sources = [''.join(c['source']) for c in nb['cells']]
            self.assertTrue(all(s in new_sources for s in original_non_engine))
            for source in new_sources:
                if source.startswith('%%writefile v51_') or source.startswith('%%writefile sage.py'):
                    compile(source.split('\n', 1)[1], 'embedded.py', 'exec')
            with self.assertRaisesRegex(ValueError, 'exists'):
                B.build(folder / 'decision.json', folder / 'bundle', 'testuser')

    def test_reject_empty_component_build(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'decision.json'
            dump(p, dict(method='v51_eb_sibling_experts', enabled=dict(eb=False, experts=[], build=False),
                         eb=dict(ci95=[-.1, .1]), groups={}))
            with self.assertRaisesRegex(ValueError, 'No accepted'):
                B.build(p, Path(t) / 'out', 'testuser')
            self.assertFalse((Path(t) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
