"""Build the frozen public Notebook with validation-approved 11/12 native states."""
import copy, glob, hashlib, json, shutil, sys
from pathlib import Path
from shared import ROOT as R, BASE
SRC = BASE / 'external/pinned_notebook'
KERNEL, DS = None, None
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12: StyleRank 11/12 native fold states (our trainer, public native layout) in the native slot; validation-gated
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
for _slot, (_src, _j) in sorted(_ds['slots'].items()):
    _t = f'{NATIVE_PUB}/fold{_j}/trainables.safetensors' if _src == 'public' else f'{_DYL}/dyl11_12_fold{_j}_trainables.safetensors'
    _c = f'{NATIVE_PUB}/calibration/fold{_j}.json' if _src == 'public' else f'{_DYL}/dyl11_12_fold{_j}_calibration.json'
    os.makedirs(f'{NATIVE}/fold{_slot}', exist_ok=True)
    if not os.path.exists(f'{NATIVE}/fold{_slot}/trainables.safetensors'):
        os.symlink(_t, f'{NATIVE}/fold{_slot}/trainables.safetensors')
    shutil.copyfile(_c, f'{NATIVE}/calibration/fold{_slot}.json')
    assert _shapes(_t) == _ref, _slot
    _q = json.load(open(f'{NATIVE}/calibration/fold{_slot}.json'))['models']['qwen8']
    assert not _q['outer_labels_used'] and _q['centered_head_trained'] and _q['plain_scale'] > 0 and _q['centered_scale'] > 0 and 0 <= _q['selected_centered_weight'] <= 0.5, _slot
assert all(os.path.exists(f'{NATIVE}/fold{k}/trainables.safetensors') and os.path.exists(f'{NATIVE}/calibration/fold{k}.json') for k in range(5)), NATIVE
print('DYL_11_12_NATIVE_MIX OK', dict(sorted(_ds['slots'].items())), flush=True)'''


def slots(decision, used):
    """slot -> (source, fold): our folds first (native processes slots in order), public folds fill the rest."""
    s = {str(i): ('ours', int(k)) for i, k in enumerate(used)}
    for i in range(len(used), 5):
        s[str(i)] = ('public', i)
    return s


def build(decision_path, runs, out, username):
    global KERNEL, DS
    if not username or '/' in username:
        raise ValueError('Provide your Kaggle username')
    KERNEL, DS = f'{username}/ntl-11-12', f'{username}/ntl-11-12-native'
    D = json.load(open(decision_path)); dec = D['decision']; used = D.get('our_folds_used', [])
    assert dec in ('public_only', 'ours2', 'ours4') and len(used) == {'public_only': 0, 'ours2': 2, 'ours4': 4}[dec]
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'; sha = hashlib.sha256(src_nb.read_bytes()).hexdigest()
    assert SRC_SHA.startswith('@') or sha == SRC_SHA, 'pinned public notebook changed'
    nb = json.load(open(src_nb)); new = copy.deepcopy(nb)
    md = (f'## 11/12 — decision `{dec}` (validation-gated, see README.md)\n'
          + ('Public 0.684 pipeline unchanged.' if dec == 'public_only' else
             f'Native slots: {json.dumps(slots(dec, used))}. Our StyleRank 11/12 fold states (Qwen3-Embedding-8B, public native layout, '
             f'our trainer) replace public native fold states; everything else unchanged.')
          + f"\nValidation (held-out official-train rows, proxy stack): {json.dumps(D.get('candidates', {}))}")
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    if dec != 'public_only':
        cell = new['cells'][3]; src = ''.join(cell['source']); assert src.count(ANCHOR) == 1, 'anchor not found once'
        cell['source'] = [l for l in (src.replace(ANCHOR, ANCHOR + PATCH)).splitlines(keepends=True)]
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    json.dump(new, open(out / 'final/submission.ipynb', 'w'), indent=1, ensure_ascii=False)
    meta = json.load(open(SRC / 'kernel-metadata.json')); meta.pop('id_no', None)
    meta.update(id=KERNEL, title='11/12', code_file='submission.ipynb', is_private=True)
    if dec != 'public_only':
        meta['dataset_sources'] = list(meta['dataset_sources']) + [DS]
    json.dump(meta, open(out / 'final/kernel-metadata.json', 'w'), indent=1)
    up = out / 'upload'; shutil.rmtree(up, ignore_errors=True)
    if dec != 'public_only':
        up.mkdir()
        for k in used:
            shutil.copy(Path(runs) / f'fold{k}/trainables.safetensors', up / f'dyl11_12_fold{k}_trainables.safetensors')
            shutil.copy(Path(runs) / f'fold{k}/calibration/fold.json', up / f'dyl11_12_fold{k}_calibration.json')
        json.dump({'slots': slots(dec, used), 'decision': dec}, open(up / 'dyl_11_12_native.json', 'w'))
        json.dump({'title': '11/12 native states', 'id': DS, 'licenses': [{'name': 'CC-BY-SA-4.0'}]}, open(up / 'dataset-metadata.json', 'w'))
    return dec, used, sha


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build a private Kaggle Notebook; no upload or submission')
    p.add_argument('--decision', type=Path, default=R / 'runs/decision.json')
    p.add_argument('--runs', type=Path, default=R / 'runs')
    p.add_argument('--out', type=Path, default=R / 'submission')
    p.add_argument('--username', required=True)
    a = p.parse_args()
    print(build(a.decision, a.runs, a.out, a.username))
