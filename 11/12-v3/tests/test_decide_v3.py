"""Gate behaviour on the real pinned OOF with synthetic candidate outputs (CPU, no model):
identical states -> public_only; a v3 state that helps -> v3_wise050; v3 harmful but v2 helpful -> v2 fallback.
Also builds the Notebook from the chosen plan and checks that only the setup cell differs from the pinned one."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared  # noqa: F401  (sets the 11/11, v1, v2 import paths)
import common as C
import decide_v3 as D
import build_submission as B
from oof_targets import sha


def family(root, kind, change, seed):
    pp, pc, _ = C.public_native_oof()
    rng = np.random.default_rng(seed)
    for k in range(5):
        run = root / f'fold{k}'; run.mkdir(parents=True)
        val = C.FOLDS[k]
        y = C.y_all[val]
        out = dict(val=val)
        for variant, scale in (('public', 0.), ('ours', 1.), ('wise050', .5)):
            plain, cent = pp[val].copy(), pc[val].copy()
            pick = rng.uniform(size=len(val)) < .4
            target = y if change == 'help' else (y + 1) % 18
            if change != 'none':
                plain[pick, target[pick]] += 4 * scale; cent[pick, target[pick]] += 4 * scale
            out[f'plain_{variant}'], out[f'centered_{variant}'], out[f'bias_{variant}'] = plain, cent, C.public_bias(k)
        np.savez(run / 'native_eval.npz', **out)
        for name in ('trainables.safetensors', 'trainables_wise050.safetensors'):
            (run / name).write_bytes(f'{kind}{k}{name}'.encode())
        np.save(run / 'val_rows.npy', val)
        rec = dict(fold=k, parity_ok=True, artifact_sha256={n: sha(run / n) for n in
                   ['trainables.safetensors', 'trainables_wise050.safetensors', 'native_eval.npz', 'val_rows.npy']})
        (run / 'native_eval.json').write_text(json.dumps(rec))


def decide(root, v3, v2):
    argv = sys.argv[:]
    sys.argv = ['decide_v3.py', '--runs', str(v3), '--fallback', str(v2), '--out', str(root / 'decision.json')]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            D.main()
    finally:
        sys.argv = argv
    return json.loads((root / 'decision.json').read_text())


def test(root):
    results = {}
    for case, (v3_change, v2_change, expected) in enumerate([('none', 'none', 'public_only'),
                                                             ('help', 'none', 'v3_wise050'),
                                                             ('harm', 'help', 'v2_wise050')]):
        base = root / f'case{case}'
        family(base / 'v3', 'v3', v3_change, 10 + case); family(base / 'v2', 'v2', v2_change, 20 + case)
        out = decide(base, base / 'v3', base / 'v2')
        assert out['decision'] == expected, (case, out['decision'], {k: (v['delta'], v['ci95'], v['passed']) for k, v in out['candidates'].items()})
        results[expected] = {k: round(v['delta'], 2) for k, v in out['candidates'].items()}
        if case == 0:
            assert all(abs(v['delta']) < 1e-9 for v in out['candidates'].values())
            assert abs(out['proxy']['public_proxy_ba'] - 66.1) < 1.0, out['proxy']['public_proxy_ba']
        if expected != 'public_only':
            plan = B.build(base / 'decision.json', base / 'submission', 'reproducer')
            nb = json.loads(Path(plan['notebook']).read_text())
            pinned = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
            assert len(nb['cells']) == len(pinned['cells']) + 1
            diff = [i for i, (a, b) in enumerate(zip(pinned['cells'], nb['cells'][1:])) if a['source'] != b['source']]
            assert diff == [2] and B.PATCH in ''.join(nb['cells'][3]['source'])
            compile(''.join(nb['cells'][3]['source']), 'setup-cell', 'exec')
            spec = json.loads((base / 'submission/upload' / B.SPEC_NAME).read_text())
            assert spec['candidate'] == expected and len(spec['sha256']) == 5
            meta = json.loads((base / 'submission/final/kernel-metadata.json').read_text())
            assert meta['id'] == 'reproducer/ntl-11-12-v3' and meta['is_private'] and not meta['enable_internet']
            assert 'reproducer/ntl-11-12-v3-native' in meta['dataset_sources']
    print(json.dumps(dict(gate_cases=results, notebook_only_setup_cell_changed=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v3_decide_') as d:
        test(Path(d))
