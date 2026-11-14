"""CPU: (1) the rewritten weights reproduce z' = (I + D) z + d exactly for any feature vector (real public heads);
(2) the notebook setup patch, run offline on the REAL public fold files, writes merged states that native_q8 would
load (same keys/shapes/dtypes), with the four heads replaced and every other tensor bit-identical; a tampered head
file is rejected; (3) the built notebook differs from the pinned one only in the setup cell and two infer.py lines."""
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared  # noqa: F401
import common as C
import neuron_rewrite as N
import build_submission as B

ROOT = Path(__file__).resolve().parents[1]


def test(tmp):
    rec = json.loads((ROOT / 'artifacts/snr_edit.json').read_text())
    edit = N.from_json(rec['edit'])
    pub = load_file(str(C.NW / 'fold2/trainables.safetensors'))
    heads = load_file(str(ROOT / 'upload/snr_fold2_heads.safetensors'))
    f = np.random.default_rng(0).normal(size=(7, 32768))
    for h in ('plain', 'centered'):
        z = f @ pub[f'{h}.weight'].T.astype(np.float64) + pub[f'{h}.bias']
        z2 = f @ heads[f'{h}.weight'].T.astype(np.float64) + heads[f'{h}.bias']
        assert np.abs(N.apply(z, *edit[h]) - z2).max() < 1e-3, h
    # offline setup patch on the real public files
    native = tmp / 'native'; (native / 'calibration').mkdir(parents=True)
    for name in ('manifest.json', 'base'):
        os.symlink(C.NW / name, native / name)
    for k in range(5):
        os.symlink(C.NW / f'fold{k}', native / f'fold{k}')
        shutil.copy(C.NW / f'calibration/fold{k}.json', native / f'calibration/fold{k}.json')
    inp = tmp / 'input/ours'; shutil.copytree(ROOT / 'upload', inp)
    code = B.PATCH.replace('/kaggle/input', str(tmp / 'input')).replace('/kaggle/temp/dyl_native_v5_snr', str(tmp / 'mix'))
    exec(compile(code, 'setup', 'exec'), dict(NATIVE=str(native), glob=glob, os=os, json=json, shutil=shutil))
    for k in range(5):
        merged = load_file(str(tmp / f'mix/fold{k}/trainables.safetensors'))
        public = load_file(str(C.NW / f'fold{k}/trainables.safetensors'))
        ours = load_file(str(ROOT / f'upload/snr_fold{k}_heads.safetensors'))
        assert set(merged) == set(public)
        for name, value in merged.items():
            target = ours.get(name, public[name])
            assert value.dtype == np.float32 and value.shape == public[name].shape and np.array_equal(value, target), (k, name)
    bad = inp / 'snr_fold3_heads.safetensors'
    t = load_file(str(bad)); t['plain.bias'] = t['plain.bias'] + 1; save_file(t, str(bad))
    try:
        exec(compile(B.PATCH.replace('/kaggle/input', str(tmp / 'input')).replace('/kaggle/temp/dyl_native_v5_snr', str(tmp / 'mix2')),
                     'setup', 'exec'), dict(NATIVE=str(native), glob=glob, os=os, json=json, shutil=shutil))
    except AssertionError:
        pass
    else:
        raise AssertionError('A tampered head file was accepted')
    plan = B.build(ROOT / 'upload', tmp / 'build', 'reproducer', {'x': 1})
    nb = json.loads(Path(plan['notebook']).read_text())
    pinned = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
    src = lambda c: ''.join(c['source'])
    engine = next(i for i, c in enumerate(pinned['cells']) if src(c).startswith('%%writefile infer.py'))
    diff = [i for i, (a, b) in enumerate(zip(pinned['cells'], nb['cells'][1:])) if src(a) != src(b)]
    assert diff == [2, engine]
    before, after = src(pinned['cells'][engine]).splitlines(), src(nb['cells'][engine + 1]).splitlines()
    assert len(before) == len(after) and sum(x != y for x, y in zip(before, after)) == 2
    compile('\n'.join(after[1:]), 'infer.py', 'exec')
    print(json.dumps(dict(weights_reproduce_logit_edit=True, setup_patch_real_files=True, other_tensors_bit_identical=True,
                          tampered_head_rejected=True, notebook_minimal_diff=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v5_snr_', dir='/data/dyl/name-that-llm-jnu/grpo_v2_review') as d:
        test(Path(d))
