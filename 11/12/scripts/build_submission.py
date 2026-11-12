"""Build the frozen public Notebook with validation-approved 11/12 native states."""
import copy, glob, hashlib, json, shutil, sys
from pathlib import Path
from shared import ROOT as R, BASE
SRC = BASE / 'external/pinned_notebook'
KERNEL, DS = None, None
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12: public-native GRPO fold states in the public ensemble
_DYL_SPEC = sorted(glob.glob('/kaggle/input/**/dyl_11_12_native.json', recursive=True))
assert len(_DYL_SPEC) == 1, _DYL_SPEC
_DYL = os.path.dirname(_DYL_SPEC[0]); _ds = json.load(open(_DYL_SPEC[0]))
NATIVE_PUB, NATIVE = NATIVE, '/kaggle/temp/dyl_native_mix'
os.makedirs(f'{NATIVE}/calibration', exist_ok=True)
for _n in ('manifest.json', 'base'):
    if not os.path.exists(f'{NATIVE}/{_n}'):
        os.symlink(f'{NATIVE_PUB}/{_n}', f'{NATIVE}/{_n}')
from safetensors import safe_open
def _shapes(_p):
    with safe_open(_p, 'np') as _f:
        return {_k: tuple(_f.get_slice(_k).get_shape()) for _k in _f.keys()}
_ref = _shapes(f'{NATIVE_PUB}/fold0/trainables.safetensors')
if _ds.get('method') == 'public_native_grpo':
    import hashlib
    for _filename, _expected in _ds['sha256'].items():
        _hash = hashlib.sha256()
        with open(f'{_DYL}/{_filename}', 'rb') as _stream:
            for _block in iter(lambda: _stream.read(8 << 20), b''):
                _hash.update(_block)
        assert _hash.hexdigest() == _expected, _filename
for _slot, (_src, _j) in sorted(_ds['slots'].items()):
    _t = f'{NATIVE_PUB}/fold{_j}/trainables.safetensors' if _src == 'public' else f'{_DYL}/dyl11_12_fold{_j}_trainables.safetensors'
    _c = f'{NATIVE_PUB}/calibration/fold{_j}.json' if _src == 'public' else f'{_DYL}/dyl11_12_fold{_j}_calibration.json'
    os.makedirs(f'{NATIVE}/fold{_slot}', exist_ok=True)
    _link = f'{NATIVE}/fold{_slot}/trainables.safetensors'
    if os.path.lexists(_link):
        assert os.path.islink(_link), _link
        os.unlink(_link)
    os.symlink(_t, _link)
    assert os.path.realpath(_link) == os.path.realpath(_t), _slot
    shutil.copyfile(_c, f'{NATIVE}/calibration/fold{_slot}.json')
    assert _shapes(_t) == _ref, _slot
    _q = json.load(open(f'{NATIVE}/calibration/fold{_slot}.json'))['models']['qwen8']
    assert not _q['outer_labels_used'] and _q['centered_head_trained'] and _q['plain_scale'] > 0 and _q['centered_scale'] > 0 and 0 <= _q['selected_centered_weight'] <= 0.5, _slot
assert all(os.path.exists(f'{NATIVE}/fold{k}/trainables.safetensors') and os.path.exists(f'{NATIVE}/calibration/fold{k}.json') for k in range(5)), NATIVE
print('DYL_11_12_NATIVE_MIX OK', dict(sorted(_ds['slots'].items())), flush=True)
if _ds.get('method') == 'public_native_grpo':
    assert sorted(int(_j) for _src, _j in _ds['slots'].values() if _src == 'ours') == sorted(_ds['optimized_folds'])
    print('DYL_GRPO_FOLDS_LOADED', _ds['optimized_folds'], flush=True)'''


def slots(decision, used):
    """slot -> (source, fold): our folds first (native processes slots in order), public folds fill the rest."""
    s = {str(i): ('ours', int(k)) for i, k in enumerate(used)}
    for i, fold in enumerate((k for k in range(5) if k not in used), len(used)):
        s[str(i)] = ('public', fold)
    return s


def strict_native_inference(source):
    """Keep the public ensemble math; require its optimized native stage to run."""
    replacements = {
        "    folds = z['folds'].tolist()":
            "    folds = z['folds'].tolist()\n"
            "    assert code == 0 and sorted(folds) == list(range(5)), 'GRPO requires all five native folds'\n"
            "    assert z['finite'].all(), 'GRPO native predictions contain invalid rows'",
        "        log('NATIVE SKIPPED: not enough time left')":
            "        raise RuntimeError('Insufficient time for required GRPO native inference')",
        "            log('NATIVE SKIPPED:', repr(_e))":
            "            raise RuntimeError('Required GRPO native inference failed') from _e",
        "    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)\n    return probs":
            "    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)\n"
            "    log('DYL_GRPO_NATIVE_PREDICTED', folds)\n    return probs",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('Pinned public native inference anchor changed')
        source = source.replace(before, after)
    return source


def build(decision_path, runs, out, username):
    global KERNEL, DS
    if not username or '/' in username:
        raise ValueError('Provide your Kaggle username')
    KERNEL, DS = f'{username}/ntl-11-12-grpo', f'{username}/ntl-11-12-grpo-native'
    D = json.load(open(decision_path)); dec = D['decision']; used = D.get('our_folds_used', [])
    if (dec != 'grpo' or D.get('method') != 'public_native_grpo' or not used
            or len(set(used)) != len(used) or not set(used) <= set(range(5))):
        raise ValueError('Submission requires completed public-native GRPO fold states')
    if dec == 'grpo':
        from windows import sha as file_sha
        for k in used:
            root = Path(runs) / f'fold{k}'
            check = D['artifacts'][str(k)]
            for key, filename in [('done', 'done.json'), ('evaluation', 'native_eval.json'),
                                  ('state', 'trainables.safetensors'), ('calibration', 'calibration/fold.json')]:
                if file_sha(root / filename) != check[key]:
                    raise ValueError(f'GRPO build plan is stale: fold{k}/{filename}')
            done = json.loads((root / 'done.json').read_text())
            if done.get('optimizer_steps', 0) < 1 or check['state'] == done['public_sha256']['state']:
                raise ValueError('GRPO needs actual updates to the public weights')
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'; sha = hashlib.sha256(src_nb.read_bytes()).hexdigest()
    assert SRC_SHA.startswith('@') or sha == SRC_SHA, 'pinned public notebook changed'
    nb = json.load(open(src_nb)); new = copy.deepcopy(nb)
    md = (f'## 11/12 — optimized native fold states `{dec}`\n'
          + f'Native slots: {json.dumps(slots(dec, used))}. Optimized Qwen3-Embedding-8B states are loaded into the public ensemble.\n'
          + f"Validation (held-out official-train rows, proxy stack): {json.dumps(D.get('candidates', {}))}")
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    cell = new['cells'][3]; src = ''.join(cell['source']); assert src.count(ANCHOR) == 1, 'anchor not found once'
    cell['source'] = [l for l in (src.replace(ANCHOR, ANCHOR + PATCH)).splitlines(keepends=True)]
    inference_cells = [c for c in new['cells'] if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
    if len(inference_cells) != 1:
        raise ValueError('Expected exactly one public inference engine')
    inference_cells[0]['source'] = strict_native_inference(''.join(inference_cells[0]['source'])).splitlines(keepends=True)
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    json.dump(new, open(out / 'final/submission.ipynb', 'w'), indent=1, ensure_ascii=False)
    meta = json.load(open(SRC / 'kernel-metadata.json')); meta.pop('id_no', None)
    meta.update(id=KERNEL, title='11/12 GRPO public continuation', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [DS]
    json.dump(meta, open(out / 'final/kernel-metadata.json', 'w'), indent=1)
    up = out / 'upload'; shutil.rmtree(up, ignore_errors=True)
    up.mkdir()
    for k in used:
        shutil.copy(Path(runs) / f'fold{k}/trainables.safetensors', up / f'dyl11_12_fold{k}_trainables.safetensors')
        shutil.copy(Path(runs) / f'fold{k}/calibration/fold.json', up / f'dyl11_12_fold{k}_calibration.json')
    spec = {'slots': slots(dec, used), 'decision': dec}
    if dec == 'grpo':
        from windows import sha as file_sha
        spec.update(method='public_native_grpo', optimized_folds=used,
                    sha256={p.name: file_sha(p) for p in up.iterdir() if p.is_file()})
    json.dump(spec, open(up / 'dyl_11_12_native.json', 'w'))
    json.dump({'title': '11/12 native states', 'id': DS, 'licenses': [{'name': 'CC-BY-SA-4.0'}]}, open(up / 'dataset-metadata.json', 'w'))
    return dec, used, sha


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build a private Kaggle Notebook; no upload or submission')
    p.add_argument('--decision', type=Path, default=R / 'runs_grpo/decision.json')
    p.add_argument('--runs', type=Path, default=R / 'runs_grpo')
    p.add_argument('--out', type=Path, default=R / 'submission_grpo')
    p.add_argument('--username', required=True)
    a = p.parse_args()
    print(build(a.decision, a.runs, a.out, a.username))
