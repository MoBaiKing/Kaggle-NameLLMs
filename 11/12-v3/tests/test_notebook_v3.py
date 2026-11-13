"""Execute the injected setup code offline: hashes verified, five slots linked to our files, public calibration
copied, shapes checked; a tampered file or a wrong shape stops the Notebook before any inference."""
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_submission as B
from oof_targets import sha


def run_patch(upload, public, mix):
    code = B.PATCH.replace("/kaggle/input/**/dyl_11_12_v3_native.json", str(upload / B.SPEC_NAME))
    code = code.replace('/kaggle/temp/dyl_native_v3_mix', str(mix))
    scope = dict(NATIVE=str(public), glob=glob, os=os, json=json, shutil=shutil)
    exec(compile(code, '<v3-native-setup>', 'exec'), scope)
    return scope


def expect_failure(upload, public, mix):
    try:
        run_patch(upload, public, mix)
    except AssertionError:
        return
    raise AssertionError('A broken package was accepted')


def test(root):
    public, upload = root / 'public', root / 'upload'
    (public / 'base').mkdir(parents=True); (public / 'calibration').mkdir(); upload.mkdir()
    (public / 'manifest.json').write_text('{}')
    for k in range(5):
        (public / f'fold{k}').mkdir()
        save_file({'plain.weight': torch.zeros(18, 4)}, public / f'fold{k}/trainables.safetensors')
        (public / f'calibration/fold{k}.json').write_text(json.dumps({'fold': k}))
        save_file({'plain.weight': torch.full((18, 4), float(k))}, upload / f'dyl11_12_v3_fold{k}_trainables.safetensors')
    spec = dict(method='public_native_oofce_v3', candidate='v3_wise050', slots={str(k): k for k in range(5)},
                sha256={p.name: sha(p) for p in upload.iterdir()})
    (upload / B.SPEC_NAME).write_text(json.dumps(spec))
    scope = run_patch(upload, public, root / 'mix')
    mix = Path(scope['NATIVE'])
    for k in range(5):
        assert os.path.realpath(mix / f'fold{k}/trainables.safetensors') == str((upload / f'dyl11_12_v3_fold{k}_trainables.safetensors').resolve())
        assert json.loads((mix / f'calibration/fold{k}.json').read_text()) == {'fold': k}
    assert scope['NATIVE_PUB'] == str(public)
    # tampered bytes
    save_file({'plain.weight': torch.ones(18, 4)}, upload / 'dyl11_12_v3_fold2_trainables.safetensors')
    expect_failure(upload, public, root / 'mix_bad_hash')
    # wrong shape with a matching hash
    save_file({'plain.weight': torch.ones(18, 5)}, upload / 'dyl11_12_v3_fold2_trainables.safetensors')
    spec['sha256']['dyl11_12_v3_fold2_trainables.safetensors'] = sha(upload / 'dyl11_12_v3_fold2_trainables.safetensors')
    (upload / B.SPEC_NAME).write_text(json.dumps(spec))
    expect_failure(upload, public, root / 'mix_bad_shape')
    print(json.dumps(dict(setup_patch_executed=True, five_slots_linked=True, public_calibration=True,
                          tampered_file_rejected=True, wrong_shape_rejected=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v3_notebook_') as d:
        test(Path(d))
