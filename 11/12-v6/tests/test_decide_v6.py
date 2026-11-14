"""Gate + build on the real pinned OOF (CPU, no model): synthetic fold evaluations equal to the public native ->
not_ready (no build); a helpful perturbation -> v6 and a Notebook that differs from the pinned one only in the setup
cell and the two native fallbacks; the setup patch runs offline and rejects a tampered fold file; a fold missing
parity -> not_ready."""
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared  # noqa: F401
import common as C
import decide_v6 as D
import build_submission as B
from windows import sha


def family(root, helpful, seed):
    pp, pc, _ = C.public_native_oof(); rng = np.random.default_rng(seed)
    for k in range(5):
        run = root / f'fold{k}'; (run / 'calibration').mkdir(parents=True)
        val = C.FOLDS[k]; plain, cent = pp[val].copy(), pc[val].copy()
        if helpful:
            pick = rng.uniform(size=len(val)) < .25
            plain[pick, C.y_all[val][pick]] += 3; cent[pick, C.y_all[val][pick]] += 3
        np.savez(run / 'native_eval.npz', val=val, plain_ours=plain, centered_ours=cent, bias_ours=C.public_bias(k))
        (run / 'calibration/fold.json').write_text(json.dumps({'models': {'qwen8': C.public_state(k)}}))
        save_file({'plain.weight': torch.full((18, 4), float(k + 1 + 10 * helpful))}, run / 'trainables.safetensors')
        rec = dict(parity_ok=True, public_runtime_argmax_agreement=.99, export_argmax_agreement_single_window=.99,
                   artifact_sha256={n: sha(run / n) for n in ['trainables.safetensors', 'native_eval.npz', 'calibration/fold.json']})
        (run / 'native_eval.json').write_text(json.dumps(rec))


def decide(runs):
    argv = sys.argv[:]; sys.argv = ['decide_v6.py', '--runs', str(runs)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            D.main()
    finally:
        sys.argv = argv
    return json.loads((runs / 'decision.json').read_text())


def test(root):
    family(root / 'same', False, 0); family(root / 'good', True, 1)
    same, good = decide(root / 'same'), decide(root / 'good')
    assert same['decision'] == 'not_ready' and abs(same['native_alone']['groups_max2']['delta']) < 1e-9
    assert good['decision'] == 'v6' and good['native_alone']['groups_max2']['ci95'][0] > 0
    try:
        B.build(root / 'same/decision.json', root / 'b_same', 'reproducer')
    except SystemExit:
        pass
    else:
        raise AssertionError('A failed gate was built')
    plan = B.build(root / 'good/decision.json', root / 'b_good', 'reproducer')
    nb = json.loads(Path(plan['notebook']).read_text()); pinned = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
    src = lambda c: ''.join(c['source'])
    engine = next(i for i, c in enumerate(pinned['cells']) if src(c).startswith('%%writefile infer.py'))
    assert [i for i, (x, y) in enumerate(zip(pinned['cells'], nb['cells'][1:])) if src(x) != src(y)] == [2, engine]
    before, after = src(pinned['cells'][engine]).splitlines(), src(nb['cells'][engine + 1]).splitlines()
    assert sum(x != y for x, y in zip(before, after)) == 2 and len(before) == len(after)
    compile('\n'.join(after[1:]), 'infer.py', 'exec')
    # offline setup patch with mock public assets
    pub = root / 'pub'; (pub / 'base').mkdir(parents=True); (pub / 'calibration').mkdir(); (pub / 'manifest.json').write_text('{}')
    for k in range(5):
        (pub / f'fold{k}').mkdir(); save_file({'plain.weight': torch.zeros(18, 4)}, pub / f'fold{k}/trainables.safetensors')
    shutil.copytree(root / 'b_good/upload', root / 'input/ours')
    code = lambda mix: B.PATCH.replace('/kaggle/input', str(root / 'input')).replace('/kaggle/temp/dyl_native_v6', str(root / mix))
    exec(compile(code('mix'), 'setup', 'exec'), dict(NATIVE=str(pub), glob=glob, os=os, json=json, shutil=shutil))
    for k in range(5):
        assert os.path.realpath(root / f'mix/fold{k}/trainables.safetensors') == str((root / f'input/ours/v6_fold{k}_trainables.safetensors').resolve())
    save_file({'plain.weight': torch.ones(18, 4)}, root / 'input/ours/v6_fold2_trainables.safetensors')
    try:
        exec(compile(code('mix2'), 'setup', 'exec'), dict(NATIVE=str(pub), glob=glob, os=os, json=json, shutil=shutil))
    except AssertionError:
        pass
    else:
        raise AssertionError('Tampered fold accepted')
    bad = json.loads((root / 'good/fold3/native_eval.json').read_text()); bad['parity_ok'] = False
    (root / 'good/fold3/native_eval.json').write_text(json.dumps(bad))
    assert decide(root / 'good')['decision'] == 'not_ready'
    print(json.dumps(dict(identical_not_ready=True, helpful_passes=dict(delta=round(good['native_alone']['groups_max2']['delta'], 2)),
                          failed_gate_not_built=True, notebook_minimal_diff=True, setup_patch_offline=True,
                          tampered_rejected=True, parity_fail_not_ready=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v6_decide_') as d:
        test(Path(d))
