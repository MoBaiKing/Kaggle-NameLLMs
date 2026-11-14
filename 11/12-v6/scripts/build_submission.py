"""Private Kaggle Notebook for v6: the pinned public 0.684 notebook with the five native fold states AND their
calibration files replaced by v6's (same architecture, so runtime and memory are unchanged). infer.py: only the two
NATIVE SKIPPED fallbacks change (delete the blend-only output and raise; no public fallback). The builder asserts
nothing else changes and refuses a decision that did not pass the gate. No upload or submission here."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

from shared import BASE, ROOT
from windows import sha

SRC = BASE / 'external/pinned_notebook'
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'
SPEC = 'dyl_v6_native.json'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12-v6: native fold states retrained on Open Pool v15 (+ seed soup); public fusion code unchanged
_DYL_SPEC = sorted(glob.glob('/kaggle/input/**/dyl_v6_native.json', recursive=True))
assert len(_DYL_SPEC) == 1, _DYL_SPEC
_DYL = os.path.dirname(_DYL_SPEC[0]); _ds = json.load(open(_DYL_SPEC[0]))
assert _ds['method'] == 'v6_native_data_soup' and sorted(_ds['folds']) == [str(k) for k in range(5)], _ds
NATIVE_PUB, NATIVE = NATIVE, '/kaggle/temp/dyl_native_v6'
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
for _k in range(5):
    _t, _c = f'{_DYL}/v6_fold{_k}_trainables.safetensors', f'{_DYL}/v6_fold{_k}_calibration.json'
    assert _sha(_t) == _ds['folds'][str(_k)]['trainables_sha256'] and _sha(_c) == _ds['folds'][str(_k)]['calibration_sha256'], _k
    assert _shapes(_t) == _ref, _k
    os.makedirs(f'{NATIVE}/fold{_k}', exist_ok=True)
    _link = f'{NATIVE}/fold{_k}/trainables.safetensors'
    if os.path.lexists(_link):
        os.unlink(_link)
    os.symlink(_t, _link)
    shutil.copyfile(_c, f'{NATIVE}/calibration/fold{_k}.json')
    _q = json.load(open(f'{NATIVE}/calibration/fold{_k}.json'))['models']['qwen8']
    assert not _q['outer_labels_used'] and _q['centered_head_trained'] and _q['plain_scale'] > 0 and _q['centered_scale'] > 0 and 0 <= _q['selected_centered_weight'] <= 0.5, _k
print('DYL_V6_NATIVE_READY OK', flush=True)'''
STRICT = {
    "        log('NATIVE SKIPPED: not enough time left')":
        "        os.path.exists(a.out) and os.remove(a.out); raise RuntimeError('Our native weights could not run in time; refusing a public-blend fallback')",
    "            log('NATIVE SKIPPED:', repr(_e))":
        "            os.path.exists(a.out) and os.remove(a.out); raise RuntimeError('Our native stage failed; refusing a public-blend fallback') from _e",
}


def build(decision_path, out, username):
    if not username or '/' in username:
        raise ValueError('Provide your Kaggle username')
    D = json.loads(Path(decision_path).read_text())
    if D.get('decision') != 'v6' or not D['gate']['passed'] or D.get('public_fallback') is not False:
        raise SystemExit('The v6 gate did not pass: nothing to build (no public fallback)')
    kernel, dataset = f'{username}/ntl-11-12-v6', f'{username}/ntl-11-12-v6-native'
    public = json.loads((NATIVE / 'manifest.json').read_text())['file_hashes']
    root = Path(D['root']); files = {}
    for k in range(5):
        rec, run = D['folds'][str(k)], root / f'fold{k}'
        if sha(run / 'trainables.safetensors') != rec['trainables_sha256'] or sha(run / 'calibration/fold.json') != rec['calibration_sha256']:
            raise ValueError(f'Build plan is stale: fold{k}')
        if rec['trainables_sha256'] == public[f'fold{k}/trainables.safetensors']:
            raise ValueError(f'fold{k} equals the public weights')
        files[f'v6_fold{k}_trainables.safetensors'] = run / 'trainables.safetensors'
        files[f'v6_fold{k}_calibration.json'] = run / 'calibration/fold.json'
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    if hashlib.sha256(src_nb.read_bytes()).hexdigest() != SRC_SHA:
        raise ValueError('Pinned public notebook changed')
    nb = json.loads(src_nb.read_text()); new = copy.deepcopy(nb)
    ev = {k: dict(base=round(v['base_ba'], 2), ours=round(v['ba'], 2), delta=round(v['delta'], 2), ci95=[round(x, 2) for x in v['ci95']])
          for k, v in D['native_alone'].items()}
    md = ('## 11/12-v6 — native expert retrained on Open Pool v15 with the public recipe + seed soup\n'
          'Pinned public 0.684 notebook; only the five Qwen3-Embedding-8B native fold states and their calibration files are '
          'replaced. Fusion, Sinkhorn and all other members unchanged.\n'
          f'Native-alone validation (official-train held-out rows, not the leaderboard): {json.dumps(ev)}')
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    setup = new['cells'][3]; text = ''.join(setup['source'])
    if text.count(ANCHOR) != 1:
        raise ValueError('Native anchor not found exactly once')
    setup['source'] = text.replace(ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    engine = [i for i, c in enumerate(new['cells']) if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
    if len(engine) != 1:
        raise ValueError('Expected one public inference engine')
    source = ''.join(new['cells'][engine[0]]['source'])
    for before, after in STRICT.items():
        if source.count(before) != 1:
            raise ValueError('Pinned native fallback anchor changed')
        source = source.replace(before, after)
    new['cells'][engine[0]]['source'] = source.splitlines(keepends=True)
    changed = [i for i, (x, y) in enumerate(zip(nb['cells'], new['cells'][1:])) if x['source'] != y['source']]
    if changed != [2, engine[0] - 1]:
        raise ValueError(f'Unexpected notebook changes: {changed}')
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    (out / 'final/submission.ipynb').write_text(json.dumps(new, indent=1, ensure_ascii=False))
    meta = json.loads((SRC / 'kernel-metadata.json').read_text()); meta.pop('id_no', None)
    meta.update(id=kernel, title='11/12 v6 native data soup', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [dataset]
    (out / 'final/kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    up = out / 'upload'; shutil.rmtree(up, ignore_errors=True); up.mkdir()
    for name, path in files.items():
        shutil.copy(path, up / name)
    spec = dict(method='v6_native_data_soup', folds={str(k): dict(trainables_sha256=sha(up / f'v6_fold{k}_trainables.safetensors'),
                                                                   calibration_sha256=sha(up / f'v6_fold{k}_calibration.json')) for k in range(5)})
    (up / SPEC).write_text(json.dumps(spec))
    (up / 'dataset-metadata.json').write_text(json.dumps({'title': '11-12 v6 native', 'id': dataset, 'licenses': [{'name': 'CC-BY-SA-4.0'}]}))
    return dict(kernel=kernel, dataset=dataset, notebook=str(out / 'final/submission.ipynb'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, default=ROOT / 'runs_v6/decision.json')
    p.add_argument('--out', type=Path, default=ROOT / 'submission_v6')
    p.add_argument('--username', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.decision, a.out, a.username), indent=1))
