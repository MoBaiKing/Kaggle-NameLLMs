"""Build the private Kaggle Notebook: the pinned public 0.684 notebook with ONLY the five native fold weight
files swapped. infer.py (fusion, Sinkhorn, fallbacks, every other member) stays byte-identical, which the
builder asserts; there is no explicit vote and no arithmetic re-mixing. No upload or submission here."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

from shared import ROOT as R, BASE
from oof_targets import sha as file_sha

SRC = BASE / 'external/pinned_notebook'
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
SPEC_NAME = 'dyl_11_12_v3_native.json'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12-v3: continued native fold states (weights only; public fusion code unchanged)
_DYL_SPEC = sorted(glob.glob('/kaggle/input/**/dyl_11_12_v3_native.json', recursive=True))
assert len(_DYL_SPEC) == 1, _DYL_SPEC
_DYL = os.path.dirname(_DYL_SPEC[0]); _ds = json.load(open(_DYL_SPEC[0]))
assert _ds['method'] == 'public_native_oofce_v3' and sorted(int(k) for k in _ds['slots']) == list(range(5)), _ds
NATIVE_PUB, NATIVE = NATIVE, '/kaggle/temp/dyl_native_v3_mix'
os.makedirs(f'{NATIVE}/calibration', exist_ok=True)
for _n in ('manifest.json', 'base'):
    if not os.path.exists(f'{NATIVE}/{_n}'):
        os.symlink(f'{NATIVE_PUB}/{_n}', f'{NATIVE}/{_n}')
import hashlib
from safetensors import safe_open
def _shapes(_p):
    with safe_open(_p, 'np') as _f:
        return {_k: tuple(_f.get_slice(_k).get_shape()) for _k in _f.keys()}
def _sha(_p):
    _h = hashlib.sha256()
    with open(_p, 'rb') as _s:
        for _b in iter(lambda: _s.read(8 << 20), b''):
            _h.update(_b)
    return _h.hexdigest()
_ref = _shapes(f'{NATIVE_PUB}/fold0/trainables.safetensors')
for _slot, _j in sorted(_ds['slots'].items()):
    _t = f'{_DYL}/dyl11_12_v3_fold{_j}_trainables.safetensors'
    assert _sha(_t) == _ds['sha256'][os.path.basename(_t)], _t
    os.makedirs(f'{NATIVE}/fold{_slot}', exist_ok=True)
    _link = f'{NATIVE}/fold{_slot}/trainables.safetensors'
    if os.path.lexists(_link):
        assert os.path.islink(_link), _link
        os.unlink(_link)
    os.symlink(_t, _link)
    assert os.path.realpath(_link) == os.path.realpath(_t) and _shapes(_t) == _ref, _slot
    shutil.copyfile(f'{NATIVE_PUB}/calibration/fold{_j}.json', f'{NATIVE}/calibration/fold{_slot}.json')
assert all(os.path.exists(f'{NATIVE}/fold{k}/trainables.safetensors') and os.path.exists(f'{NATIVE}/calibration/fold{k}.json') for k in range(5)), NATIVE
print('DYL_11_12_V3_NATIVE_MIX OK', _ds['candidate'], dict(sorted(_ds['slots'].items())), flush=True)'''


def build(decision_path, out, username):
    if not username or '/' in username:
        raise ValueError('Provide your Kaggle username')
    kernel, dataset = f'{username}/ntl-11-12-v3', f'{username}/ntl-11-12-v3-native'
    D = json.loads(Path(decision_path).read_text())
    if D.get('method') != 'public_native_oofce_v3':
        raise ValueError('Not a v3 decision file')
    if D['decision'] == 'public_only':
        raise SystemExit('Gate chose public_only: nothing to build; the unchanged public 0.684 notebook is the result')
    chosen = D['chosen']; root = Path(chosen['root']); filename = chosen['filename']
    key = f"{chosen['variant']}_sha256"
    files = {}
    for k in range(5):
        record = chosen['folds'][str(k)]
        run = root / f'fold{k}'
        if file_sha(run / 'native_eval.json') != record['evaluation_sha256'] or file_sha(run / filename) != record[key]:
            raise ValueError(f'Build plan is stale: fold{k}')
        files[k] = run / filename
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    digest = hashlib.sha256(src_nb.read_bytes()).hexdigest()
    if digest != SRC_SHA:
        raise ValueError('Pinned public notebook changed')
    nb = json.loads(src_nb.read_text()); new = copy.deepcopy(nb)
    md = (f"## 11/12-v3 — native fold states `{D['decision']}`\n"
          'Pinned public 0.684 notebook; only the five Qwen3-Embedding-8B native fold weight files are replaced. '
          'Fusion, Sinkhorn, fallbacks and all other members are unchanged.\n'
          f"Validation proxy (official-train held-out rows, not the leaderboard): "
          f"{json.dumps({k: {m: v[m] for m in ('base_ba', 'ba', 'delta', 'ci95', 'standalone_native_delta')} for k, v in D['candidates'].items()})}")
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    cell = new['cells'][3]; src = ''.join(cell['source'])
    if src.count(ANCHOR) != 1:
        raise ValueError('Native anchor not found exactly once')
    cell['source'] = src.replace(ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    changed = [i for i, (a, b) in enumerate(zip(nb['cells'], new['cells'][1:])) if a['source'] != b['source']]
    if changed != [2]:
        raise ValueError(f'Only the setup cell may change, got {changed}')
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    (out / 'final/submission.ipynb').write_text(json.dumps(new, indent=1, ensure_ascii=False))
    meta = json.loads((SRC / 'kernel-metadata.json').read_text()); meta.pop('id_no', None)
    meta.update(id=kernel, title='11/12 v3 native continuation', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [dataset]
    (out / 'final/kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    up = out / 'upload'; shutil.rmtree(up, ignore_errors=True); up.mkdir()
    for k, path in files.items():
        shutil.copy(path, up / f'dyl11_12_v3_fold{k}_trainables.safetensors')
    spec = dict(method='public_native_oofce_v3', candidate=D['decision'], slots={str(k): k for k in range(5)},
                sha256={p.name: file_sha(p) for p in sorted(up.iterdir()) if p.suffix == '.safetensors'})
    (up / SPEC_NAME).write_text(json.dumps(spec))
    (up / 'dataset-metadata.json').write_text(json.dumps({'title': '11-12 v3 native states', 'id': dataset,
                                                         'licenses': [{'name': 'CC-BY-SA-4.0'}]}))
    return dict(decision=D['decision'], kernel=kernel, dataset=dataset, notebook=str(out / 'final/submission.ipynb'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, default=R / 'runs_v3/decision.json')
    p.add_argument('--out', type=Path, default=R / 'submission_v3')
    p.add_argument('--username', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.decision, a.out, a.username), indent=1))
