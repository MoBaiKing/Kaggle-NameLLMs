"""Decision on the real pinned OOF with synthetic candidate outputs (CPU, no model). There is no public
fallback: the decision is always our v3_wise050 once five folds are ready, with proxy evidence attached
(identical -> no support; helpful -> support; harmful -> warning, v2 only reported). Missing folds -> not ready.
Also builds the Notebook and checks that only the setup cell and the two native fallbacks differ from the pinned one."""
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
    sys.argv = ['decide_v3.py', '--runs', str(v3), '--v2-diagnostics', str(v2), '--out', str(root / 'decision.json')]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            D.main()
    finally:
        sys.argv = argv
    return json.loads((root / 'decision.json').read_text())


def test(root):
    results = {}
    for case, (v3_change, v2_change, supports, warning) in enumerate([('none', 'none', False, False),
                                                                      ('help', 'none', True, False),
                                                                      ('harm', 'help', False, True)]):
        base = root / f'case{case}'
        family(base / 'v3', 'v3', v3_change, 10 + case); family(base / 'v2', 'v2', v2_change, 20 + case)
        out = decide(base, base / 'v3', base / 'v2')
        main = out['candidates']['v3_wise050']
        assert out['decision'] == 'v3_wise050' and out['evidence']['public_fallback'] is False
        assert main['proxy_supports'] == supports and bool(main['proxy_warning']) == warning, (case, main)
        assert out['evidence']['proxy_supports'] == supports
        results[v3_change] = {k: round(v['delta'], 2) for k, v in out['candidates'].items()}
        if case == 0:
            assert all(abs(v['delta']) < 1e-9 for v in out['candidates'].values())
            assert abs(out['proxy']['public_proxy_ba'] - 66.1) < 1.0, out['proxy']['public_proxy_ba']
        plan = B.build(base / 'decision.json', base / 'submission', 'reproducer')
        nb = json.loads(Path(plan['notebook']).read_text())
        pinned = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
        assert len(nb['cells']) == len(pinned['cells']) + 1
        diff = [i for i, (a, b) in enumerate(zip(pinned['cells'], nb['cells'][1:])) if a['source'] != b['source']]
        engine = next(i for i, c in enumerate(pinned['cells']) if ''.join(c['source']).startswith('%%writefile infer.py'))
        assert diff == [2, engine] and B.PATCH in ''.join(nb['cells'][3]['source'])
        before = ''.join(pinned['cells'][engine]['source']).splitlines()
        after = ''.join(nb['cells'][engine + 1]['source']).splitlines()
        changed = [(x, z) for x, z in zip(before, after) if x != z]
        assert len(before) == len(after) and len(changed) == 2 and all('NATIVE SKIPPED' in x and 'raise RuntimeError' in z for x, z in changed)
        compile(''.join(nb['cells'][3]['source']), 'setup-cell', 'exec')
        compile('\n'.join(after[1:]), 'infer.py', 'exec')
        spec = json.loads((base / 'submission/upload' / B.SPEC_NAME).read_text())
        assert spec['candidate'] == 'v3_wise050' and len(spec['sha256']) == 5
        meta = json.loads((base / 'submission/final/kernel-metadata.json').read_text())
        assert meta['id'] == 'reproducer/ntl-11-12-v3' and meta['is_private'] and not meta['enable_internet']
        assert 'reproducer/ntl-11-12-v3-native' in meta['dataset_sources']
    # A fold file identical to the public weights is refused (no public resubmission).
    fake = root / 'fake_native'; fake.mkdir()
    manifest = json.loads((B.NATIVE / 'manifest.json').read_text())
    manifest['file_hashes']['fold3/trainables.safetensors'] = sha(root / 'case1/v3/fold3/trainables_wise050.safetensors')
    (fake / 'manifest.json').write_text(json.dumps(manifest))
    real, B.NATIVE = B.NATIVE, fake
    try:
        B.build(root / 'case1/decision.json', root / 'public_build', 'reproducer')
    except ValueError as error:
        assert 'equals the public weights' in str(error), error
    else:
        raise AssertionError('Public weights were accepted')
    finally:
        B.NATIVE = real
    # Incomplete v3 -> not ready, nothing decided.
    import shutil
    shutil.rmtree(root / 'case1/v3/fold4')
    try:
        decide(root / 'case1', root / 'case1/v3', root / 'case1/v2')
    except SystemExit as error:
        assert 'not ready' in str(error)
    else:
        raise AssertionError('Incomplete v3 produced a decision')
    print(json.dumps(dict(decision_always_ours=True, evidence_cases=results, public_weights_refused=True,
                          incomplete_not_ready=True, notebook_only_setup_and_native_fallbacks_changed=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v3_decide_') as d:
        test(Path(d))
