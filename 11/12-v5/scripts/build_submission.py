"""Private Kaggle Notebook for v5.0 SNR: the pinned public 0.684 notebook with the native heads rewritten.
setup cell: the five public native fold files are SHA256-checked against the pinned manifest, their four head
tensors replaced by ours (hash-checked), and the merged states written to /kaggle/temp; public calibration kept.
infer.py: only the two NATIVE SKIPPED fallbacks change (delete the blend-only output and raise: a run without our
native must not become a public-blend submission). The builder asserts nothing else changes. No upload here."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import shared  # noqa: F401
from shared import BASE, ROOT
from oof_targets import sha

SRC = BASE / 'external/pinned_notebook'
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
SPEC = 'dyl_v5_snr.json'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12-v5.0 SNR: sibling neuron rewrite of the native output heads (public fusion code unchanged)
_DYL_SPEC = sorted(glob.glob('/kaggle/input/**/dyl_v5_snr.json', recursive=True))
assert len(_DYL_SPEC) == 1, _DYL_SPEC
_DYL = os.path.dirname(_DYL_SPEC[0]); _ds = json.load(open(_DYL_SPEC[0]))
assert _ds['method'] == 'v5_snr_sibling_neuron_rewrite', _ds
NATIVE_PUB, NATIVE = NATIVE, '/kaggle/temp/dyl_native_v5_snr'
os.makedirs(f'{NATIVE}/calibration', exist_ok=True)
for _n in ('manifest.json', 'base'):
    if not os.path.exists(f'{NATIVE}/{_n}'):
        os.symlink(f'{NATIVE_PUB}/{_n}', f'{NATIVE}/{_n}')
import hashlib
import numpy as _np
from safetensors.numpy import load_file as _lf, save_file as _sf
def _sha(_p):
    _h = hashlib.sha256()
    with open(_p, 'rb') as _s:
        for _b in iter(lambda: _s.read(8 << 20), b''):
            _h.update(_b)
    return _h.hexdigest()
_man = json.load(open(f'{NATIVE_PUB}/manifest.json'))
for _k in range(5):
    _pub = f'{NATIVE_PUB}/fold{_k}/trainables.safetensors'
    assert _sha(_pub) == _ds['public_sha256'][str(_k)] == _man['file_hashes'][f'fold{_k}/trainables.safetensors'], _k
    _hf = f'{_DYL}/snr_fold{_k}_heads.safetensors'
    assert _sha(_hf) == _ds['heads_sha256'][str(_k)], _k
    _t, _h = _lf(_pub), _lf(_hf)
    assert set(_h) == {'plain.weight', 'plain.bias', 'centered.weight', 'centered.bias'}, sorted(_h)
    for _n, _v in _h.items():
        assert _v.shape == _t[_n].shape and _v.dtype == _t[_n].dtype == _np.float32 and _np.isfinite(_v).all(), _n
        assert not _np.array_equal(_v, _t[_n]), _n
        _t[_n] = _v
    os.makedirs(f'{NATIVE}/fold{_k}', exist_ok=True)
    _sf(_t, f'{NATIVE}/fold{_k}/trainables.safetensors')
    del _t, _h
    shutil.copyfile(f'{NATIVE_PUB}/calibration/fold{_k}.json', f'{NATIVE}/calibration/fold{_k}.json')
assert all(os.path.exists(f'{NATIVE}/fold{k}/trainables.safetensors') and os.path.exists(f'{NATIVE}/calibration/fold{k}.json') for k in range(5)), NATIVE
print('DYL_V5_SNR_READY OK', _ds['lam'], flush=True)'''
STRICT = {
    "        log('NATIVE SKIPPED: not enough time left')":
        "        os.path.exists(a.out) and os.remove(a.out); raise RuntimeError('Our native weights could not run in time; refusing a public-blend fallback')",
    "            log('NATIVE SKIPPED:', repr(_e))":
        "            os.path.exists(a.out) and os.remove(a.out); raise RuntimeError('Our native stage failed; refusing a public-blend fallback') from _e",
}


def strict(source):
    for before, after in STRICT.items():
        if source.count(before) != 1:
            raise ValueError('Pinned native fallback anchor changed')
        source = source.replace(before, after)
    return source


def build(upload, out, username, evidence):
    kernel, dataset = f'{username}/ntl-dyl-v5-snr', f'{username}/ntl-dyl-v5-snr-heads'
    spec = json.loads((Path(upload) / SPEC).read_text())
    for k in range(5):
        if sha(Path(upload) / f'snr_fold{k}_heads.safetensors') != spec['heads_sha256'][str(k)]:
            raise ValueError(f'Head file {k} changed after fitting')
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    if hashlib.sha256(src_nb.read_bytes()).hexdigest() != SRC_SHA:
        raise ValueError('Pinned public notebook changed')
    nb = json.loads(src_nb.read_text()); new = copy.deepcopy(nb)
    md = ('## 11/12-v5.0 SNR — sibling neuron rewrite of the native expert heads\n'
          'Pinned public 0.684 notebook. The 18 output neurons of both Qwen3-Embedding-8B native heads are rewritten as '
          'W\' = (I + D) W, b\' = (I + D) b + d with D restricted to sibling / confusable classes, fitted on the clean '
          'out-of-fold head outputs. Fusion, Sinkhorn and all other members unchanged.\n'
          f'Validation (official-train held-out rows, nested prompt-fold CV, not the leaderboard): {json.dumps(evidence)}')
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    setup = new['cells'][3]; text = ''.join(setup['source'])
    if text.count(ANCHOR) != 1:
        raise ValueError('Native anchor not found exactly once')
    setup['source'] = text.replace(ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    engine = [i for i, c in enumerate(new['cells']) if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
    if len(engine) != 1:
        raise ValueError('Expected one public inference engine')
    new['cells'][engine[0]]['source'] = strict(''.join(new['cells'][engine[0]]['source'])).splitlines(keepends=True)
    changed = [i for i, (a, b) in enumerate(zip(nb['cells'], new['cells'][1:])) if a['source'] != b['source']]
    if changed != [2, engine[0] - 1]:
        raise ValueError(f'Unexpected notebook changes: {changed}')
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    (out / 'final/submission.ipynb').write_text(json.dumps(new, indent=1, ensure_ascii=False))
    meta = json.loads((SRC / 'kernel-metadata.json').read_text()); meta.pop('id_no', None)
    meta.update(id=kernel, title='ntl dyl v5 snr', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [dataset]
    (out / 'final/kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    ds = out / 'dataset'; shutil.rmtree(ds, ignore_errors=True); ds.mkdir()
    for p in Path(upload).iterdir():
        shutil.copy(p, ds / p.name)
    (ds / 'dataset-metadata.json').write_text(json.dumps({'title': 'ntl dyl v5 snr heads', 'id': dataset,
                                                         'licenses': [{'name': 'CC-BY-SA-4.0'}]}))
    return dict(kernel=kernel, dataset=dataset, notebook=str(out / 'final/submission.ipynb'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--username', required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'submission_v5')
    a = p.parse_args()
    ev = json.loads((ROOT / 'artifacts/snr_evidence.json').read_text())['snr_sibling']['vs_public']
    evidence = {k: dict(delta=round(v['delta'], 2), ci95=[round(x, 2) for x in v['ci95']]) for k, v in ev.items()}
    print(json.dumps(build(ROOT / 'upload', a.out, a.username, evidence), indent=1))
