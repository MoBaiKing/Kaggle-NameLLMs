"""Build the private v4 Kaggle Notebook = A1 + public 0.684 stack + A2 + B. No upload or submission here.

Changes to the pinned public notebook, all asserted by the builder:
  * setup cell: the five native fold weight files -> A1 (HSC-Native, WiSE 0.5); A2/B bundles hash-checked; the A2
    base (Kaggle model qwen-3 1.7b-base) re-hashed against the pinned HF revision; paths exported for infer.py;
  * two new %%writefile cells before infer.py: dyl_ccl.py (= ccl_gen.py) and dyl_sage.py (= sage.py);
  * infer.py: after the public native stage writes its decision, dyl_sage.run scores the gated rows' top-4
    candidates with A2 on both GPUs and applies B; the two NATIVE SKIPPED fallbacks delete the output and raise
    (a run without our native must not become a public-blend submission). Nothing else in infer.py changes.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

from shared import ROOT as R, BASE
from oof_targets import sha as file_sha
import assets_v4

SRC = BASE / 'external/pinned_notebook'
NATIVE = BASE / 'external/native_q8/ntl-native-q8-multidepth-fivefold-0679'
SRC_SHA = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
SPEC_NAME = 'dyl_11_12_v4_bundle.json'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
PATCH = r'''
# ---- 11/12-v4: A1 native fold states + A2 / B bundles (public fusion code unchanged)
_DYL_SPEC = sorted(glob.glob('/kaggle/input/**/dyl_11_12_v4_bundle.json', recursive=True))
assert len(_DYL_SPEC) == 1, _DYL_SPEC
_DYL = os.path.dirname(_DYL_SPEC[0]); _ds = json.load(open(_DYL_SPEC[0]))
assert _ds['method'] == 'v4_a1_a2_b' and sorted(int(k) for k in _ds['slots']) == list(range(5)), _ds
NATIVE_PUB, NATIVE = NATIVE, '/kaggle/temp/dyl_native_v4_mix'
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
for _f, _h in _ds['sha256'].items():
    assert _sha(f'{_DYL}/{_f}') == _h, _f
_ref = _shapes(f'{NATIVE_PUB}/fold0/trainables.safetensors')
for _slot, _j in sorted(_ds['slots'].items()):
    _t = f'{_DYL}/dyl11_12_v4_fold{_j}_trainables.safetensors'
    os.makedirs(f'{NATIVE}/fold{_slot}', exist_ok=True)
    _link = f'{NATIVE}/fold{_slot}/trainables.safetensors'
    if os.path.lexists(_link):
        assert os.path.islink(_link), _link
        os.unlink(_link)
    os.symlink(_t, _link)
    assert os.path.realpath(_link) == os.path.realpath(_t) and _shapes(_t) == _ref, _slot
    shutil.copyfile(f'{NATIVE_PUB}/calibration/fold{_j}.json', f'{NATIVE}/calibration/fold{_slot}.json')
assert all(os.path.exists(f'{NATIVE}/fold{k}/trainables.safetensors') and os.path.exists(f'{NATIVE}/calibration/fold{k}.json') for k in range(5)), NATIVE
_A2B = [os.path.dirname(_p) for _p in glob.glob('/kaggle/input/**/model.safetensors', recursive=True)
        if os.path.getsize(_p) == _ds['a2_base_bytes'] and os.path.exists(os.path.join(os.path.dirname(_p), 'config.json'))
        and _sha(os.path.join(os.path.dirname(_p), 'config.json')) == _ds['a2_base_config_sha256']]
assert len(_A2B) >= 1 and _sha(os.path.join(_A2B[0], 'model.safetensors')) == _ds['a2_base_sha256'], _A2B
os.environ.update(DYL_V4_A2_BUNDLE=_DYL, DYL_V4_A2_BASE=_A2B[0], DYL_V4_B_BUNDLE=os.path.join(_DYL, 'b_bundle.json'))
print('DYL_11_12_V4_READY OK', dict(sorted(_ds['slots'].items())), _A2B[0], flush=True)'''
INFER_EDITS = {
    "        log('NATIVE SKIPPED: not enough time left')":
        "        os.path.exists(a.out) and os.remove(a.out)\n"
        "        raise RuntimeError('Our native weights could not run in time; refusing a public-blend fallback')",
    "            log('NATIVE SKIPPED:', repr(_e))":
        "            os.path.exists(a.out) and os.remove(a.out)\n"
        "            raise RuntimeError('Our native stage failed; refusing a public-blend fallback') from _e",
    "            log(f'native q8 at share {a.native_share}: {changed}/{len(test)} decisions differ from the blend-only decision', gpu_mem())":
        "            log(f'native q8 at share {a.native_share}: {changed}/{len(test)} decisions differ from the blend-only decision', gpu_mem())\n"
        "            try:\n"
        "                import dyl_sage\n"
        "                dyl_sage.run(probs, test, os.path.join(a.data, 'test.csv'), log, write_pred, LABELS, os.environ['DYL_V4_B_BUNDLE'],\n"
        "                             os.environ['DYL_V4_A2_BUNDLE'], os.environ['DYL_V4_A2_BASE'],\n"
        "                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dyl_ccl.py'), '/kaggle/temp/dyl_v4_a2',\n"
        "                             a.native_deadline + 1800)\n"
        "            except Exception as _de:\n"
        "                log('DYL_V4_STAGE_FAILED (A1 decision kept):', repr(_de))\n"
        "                raise SystemExit(3)",
}


def edit_infer(source):
    for before, after in INFER_EDITS.items():
        if source.count(before) != 1:
            raise ValueError(f'Pinned infer.py anchor changed: {before[:50]}')
        source = source.replace(before, after)
    return source


def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True), 'outputs': [], 'execution_count': None}


def build(decision_path, out, username):
    if not username or '/' in username:
        raise ValueError('Provide your Kaggle username')
    kernel, dataset = f'{username}/ntl-11-12-v4', f'{username}/ntl-11-12-v4-bundle'
    D = json.loads(Path(decision_path).read_text())
    if D.get('method') != 'v4_a1_a2_b' or D.get('decision') != 'v4_full' or D.get('public_fallback') is not False:
        raise ValueError('Only the v4 A1+A2+B increment is built; there is no public fallback')
    public_hashes = json.loads((NATIVE / 'manifest.json').read_text())['file_hashes']
    files = {}
    for k in range(5):
        record, run = D['a1']['folds'][str(k)], Path(D['a1']['root']) / f'fold{k}'
        if file_sha(run / 'native_eval.json') != record['evaluation_sha256'] or file_sha(run / 'trainables_wise050.safetensors') != record['wise050_sha256']:
            raise ValueError(f'Build plan is stale: A1 fold{k}')
        if record['wise050_sha256'] == public_hashes[f'fold{k}/trainables.safetensors']:
            raise ValueError(f'A1 fold{k} equals the public weights')
        files[f'dyl11_12_v4_fold{k}_trainables.safetensors'] = run / 'trainables_wise050.safetensors'
    a2 = Path(D['a2']['root'])
    a2_bundle = json.loads((a2 / 'a2_bundle.json').read_text())
    adapters = file_sha(a2 / 'adapters.safetensors')
    if adapters != a2_bundle['adapters_sha256'] or adapters != D['a2']['scores']['adapters_sha256']:
        raise ValueError('A2 adapters differ from the scored ones')
    files['adapters.safetensors'] = a2 / 'adapters.safetensors'; files['a2_bundle.json'] = a2 / 'a2_bundle.json'
    if file_sha(D['b']['bundle']) != D['b']['sha256']:
        raise ValueError('B bundle changed after the decision')
    files['b_bundle.json'] = Path(D['b']['bundle'])
    src_nb = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    if hashlib.sha256(src_nb.read_bytes()).hexdigest() != SRC_SHA:
        raise ValueError('Pinned public notebook changed')
    nb = json.loads(src_nb.read_text()); new = copy.deepcopy(nb)
    ev = {k: {m: round(v[m], 3) if isinstance(v[m], float) else v[m] for m in ('base_ba', 'ba', 'delta', 'ci95')} for k, v in D['evidence'].items()}
    md = ('## 11/12-v4 — A1 + public 0.684 stack + A2 + B\n'
          'A1: HSC-Native fold states (WiSE 0.5) in the native slot. A2: CCL-Gen class-conditional likelihood expert '
          '(Qwen3-1.7B-Base + LoRA + class prefixes) on the top-4 candidates of gated rows. B: SAGE sibling-aware gated '
          're-ranker. Public fusion, Sinkhorn and members unchanged.\n'
          f'Validation evidence (official-train held-out rows, LB-anchored proxy, not the leaderboard): {json.dumps(ev)}')
    new['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [md]})
    setup = new['cells'][3]; text = ''.join(setup['source'])
    if text.count(ANCHOR) != 1:
        raise ValueError('Native anchor not found exactly once')
    setup['source'] = text.replace(ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    engine = [i for i, c in enumerate(new['cells']) if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
    if len(engine) != 1:
        raise ValueError('Expected exactly one public inference engine')
    e = engine[0]
    new['cells'][e]['source'] = edit_infer(''.join(new['cells'][e]['source'])).splitlines(keepends=True)
    here = Path(__file__).resolve().parent
    new['cells'][e:e] = [code_cell('%%writefile dyl_ccl.py\n' + (here / 'ccl_gen.py').read_text()),
                         code_cell('%%writefile dyl_sage.py\n' + (here / 'sage.py').read_text())]
    out = Path(out); shutil.rmtree(out / 'final', ignore_errors=True); (out / 'final').mkdir(parents=True)
    (out / 'final/submission.ipynb').write_text(json.dumps(new, indent=1, ensure_ascii=False))
    meta = json.loads((SRC / 'kernel-metadata.json').read_text()); meta.pop('id_no', None)
    meta.update(id=kernel, title='11/12 v4 A1 public A2 B', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [dataset]
    if assets_v4.KAGGLE_MODEL not in meta['model_sources']:
        meta['model_sources'] = list(meta['model_sources']) + [assets_v4.KAGGLE_MODEL]
    (out / 'final/kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    up = out / 'upload'; shutil.rmtree(up, ignore_errors=True); up.mkdir()
    for name, path in files.items():
        shutil.copy(path, up / name)
    spec = dict(method='v4_a1_a2_b', slots={str(k): k for k in range(5)},
                sha256={p.name: file_sha(p) for p in sorted(up.iterdir())},
                a2_base_sha256=assets_v4.MODEL_SHA256, a2_base_bytes=assets_v4.MODEL_BYTES,
                a2_base_config_sha256=a2_bundle['base']['config.json'])
    (up / SPEC_NAME).write_text(json.dumps(spec))
    (up / 'dataset-metadata.json').write_text(json.dumps({'title': '11-12 v4 bundle', 'id': dataset,
                                                         'licenses': [{'name': 'CC-BY-SA-4.0'}]}))
    return dict(kernel=kernel, dataset=dataset, notebook=str(out / 'final/submission.ipynb'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, default=R / 'runs_v4/b/decision.json')
    p.add_argument('--out', type=Path, default=R / 'submission_v4')
    p.add_argument('--username', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.decision, a.out, a.username), indent=1))
