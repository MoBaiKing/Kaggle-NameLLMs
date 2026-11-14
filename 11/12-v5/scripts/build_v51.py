"""Build only accepted v5.1 components. No gate => no Notebook, no upload, no submission."""
import argparse
import copy
import json
import shutil
from pathlib import Path

from v51_common import BASE, ROOT, RUN, CLASSES, dump, sha
from v51_evaluate import choose_components

SPEC = 'dyl_v51_bundle.json'
SRC = BASE / 'external/pinned_notebook'
PIN = '340d29bb90e4dc60a9f8bba1bc79bdc69b0faa1bf65d04e367e78a1ed264297f'
ANCHOR = "assert os.path.exists(f'{NATIVE}/base/model.safetensors.index.json') and os.path.exists(f'{NATIVE_BASE}/model.safetensors.index.json'), (NATIVE, NATIVE_BASE)"
SUCCESS = "            log(f'native q8 at share {a.native_share}: {changed}/{len(test)} decisions differ from the blend-only decision', gpu_mem())"
CALIBRATE = "    lp = native_q8.calibrate(z['plain'][:, :, order], z['centered'][:, :, order], z['centered_bias'][:, order], Gte, states, bool(a.native_cent))"
PATCH = '''
# v5.1: only components that passed prompt-OOF gates.
_v51_specs = glob.glob('/kaggle/input/**/dyl_v51_bundle.json', recursive=True)
assert len(_v51_specs) == 1, _v51_specs
os.environ['DYL_V51_BUNDLE'] = os.path.dirname(_v51_specs[0])
import hashlib
_v51 = json.load(open(_v51_specs[0]))
assert _v51['enabled']['build'] and (_v51['enabled']['eb'] or _v51['enabled']['experts'])
def _v51_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
for _name, _expected in _v51['sha256'].items():
    assert _v51_sha(os.path.join(os.environ['DYL_V51_BUNDLE'], _name)) == _expected, _name
for _name, _expected in _v51['native_sha256'].items():
    assert _v51_sha(os.path.join(NATIVE, _name)) == _expected, _name
print('DYL_V51_READY', _v51['enabled'], flush=True)
'''


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Pinned source anchor changed: {before[:90]}')
    return source.replace(before, after)


def edit_infer(source, enabled):
    source = replace_once(source, "    folds = z['folds'].tolist()", "    folds = z['folds'].tolist()\n"
                          "    assert code == 0 and folds == list(range(5)) and z['finite'].all(), 'Incomplete native'\n"
                          "    assert np.isfinite(z['plain']).all() and np.isfinite(z['centered']).all(), 'Non-finite native'")
    if enabled['eb']:
        source = replace_once(source, '        del E_q', '        # E_q retained for accepted EB centering (no extra model forward).')
        source = replace_once(source, CALIBRATE, "    import v51_runtime\n"
                              "    lp = v51_runtime.native(z['plain'][:, :, order], z['centered'][:, :, order],\n"
                              "                            z['centered_bias'][:, order], globals().get('E_q'), Gte, states, LABELS)")
    if enabled['experts']:
        source = replace_once(source, SUCCESS, SUCCESS + "\n            import v51_runtime\n"
                              "            v51_runtime.rerank(probs, test, os.path.join(a.data, 'test.csv'), log, write_pred, LABELS,\n"
                              "                               os.path.join(a.native_work, 'native-q8-base'), '/kaggle/temp/v51_experts', a.native_deadline + 1800)")
    for before in ("        log('NATIVE SKIPPED: not enough time left')", "            log('NATIVE SKIPPED:', repr(_e))"):
        indent = before[:len(before) - len(before.lstrip())]
        source = replace_once(source, before, indent + "os.path.exists(a.out) and os.remove(a.out)\n" +
                              indent + "raise RuntimeError('Accepted v5.1 component could not finish; refusing submission')")
    return source


def validate_decision(path):
    decision = json.loads(Path(path).read_text())
    expected = choose_components(decision['eb'], decision['groups'], decision.get('combined'))
    if decision.get('method') != 'v51_eb_sibling_experts' or decision['enabled'] != expected or not expected['build']:
        raise ValueError('No accepted component or inconsistent gate decision; do not build or submit')
    if decision.get('proxy_used_for_acceptance') is not False or decision.get('hidden_test_evaluated') is not False:
        raise ValueError('Validation protocol changed')
    for file, digest in decision['artifacts'].items():
        if sha(file) != digest:
            raise ValueError(f'Stale validation artifact: {file}')
    return decision


def build(decision_path, out, username):
    if not username or '/' in username:
        raise ValueError('Provide a Kaggle username')
    D = validate_decision(decision_path); enabled = D['enabled']; evidence = Path(decision_path).parent
    original = SRC / 'ntl-684-ozertuu-mirror.ipynb'
    if sha(original) != PIN:
        raise ValueError('Pinned public Notebook changed')
    nb = json.loads(original.read_text()); new = copy.deepcopy(nb)
    def cell(text):
        return dict(cell_type='code', metadata={}, source=text.splitlines(keepends=True), outputs=[], execution_count=None)
    setup = ''.join(new['cells'][2]['source'])
    new['cells'][2]['source'] = replace_once(setup, ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    indexes = [i for i, c in enumerate(new['cells']) if ''.join(c.get('source', [])).startswith('%%writefile infer.py\n')]
    if len(indexes) != 1:
        raise ValueError('Expected one inference engine')
    e = indexes[0]
    new['cells'][e]['source'] = edit_infer(''.join(new['cells'][e]['source']), enabled).splitlines(keepends=True)
    changed = [i for i, (a, b) in enumerate(zip(nb['cells'], new['cells'])) if a['source'] != b['source']]
    if changed != [2, e]:
        raise ValueError('Unexpected public Notebook changes')
    import shared  # noqa: F401
    import sage
    files = {name: ROOT / 'scripts' / name for name in
             ('v51_common.py', 'v51_eb.py', 'v51_expert.py', 'v51_sage.py', 'v51_runtime.py')}
    files['sage.py'] = Path(sage.__file__)
    if enabled['eb']:
        files['eb_models.npz'] = evidence / 'eb_models.npz'
    if enabled['experts']:
        files['sage_bundle.json'] = evidence / 'sage_bundle.json'
        # Resolve checkpoints from verified records; do not assume the caller's --out matches training.
        for group in enabled['experts']:
            for k in range(5):
                suffix = f'/{group}/exclude_{k}/final.safetensors'
                candidates = [Path(p) for p in D['artifacts'] if p.endswith(suffix)]
                if len(candidates) != 1:
                    raise ValueError('Missing validated expert checkpoint')
                files[f'{group}_fold{k}.safetensors'] = candidates[0]
    new['cells'][e:e] = [cell('%%writefile ' + n + '\n' + p.read_text()) for n, p in files.items() if n.endswith('.py')]
    new['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=[
        '# 11/12-v5.1: empirical Bayes prompt centering + 8B discriminative sibling experts\n'
        f'Accepted components: {json.dumps(enabled)}. Evidence is official-train prompt OOF validation, '
        'not hidden-test or leaderboard performance. Public native weights and public mixture remain frozen.']))
    out = Path(out)
    if out.exists():
        raise ValueError('Build destination exists; retain prior bundles and use a new destination')
    (out / 'final').mkdir(parents=True); (out / 'upload').mkdir()
    for name, path in files.items():
        shutil.copyfile(path, out / 'upload' / name)
    files_hash = {n: sha(out / 'upload' / n) for n in files}
    kernel, dataset = f'{username}/ntl-11-12-v51', f'{username}/ntl-11-12-v51-bundle'
    assets = json.loads((BASE / 'config/assets.json').read_text())['native_assets']
    native_hashes = {'manifest.json': assets['manifest_sha256']}
    for k in range(5):
        for name in (f'fold{k}/trainables.safetensors', f'calibration/fold{k}.json'):
            native_hashes[name] = assets['files'][name]
    dump(out / 'upload' / SPEC, dict(method=D['method'], classes=CLASSES, enabled=enabled, sha256=files_hash,
         native_sha256=native_hashes,
         decision_sha256=sha(decision_path), evidence=dict(eb=D['eb'], groups=D['groups'], combined=D['combined'])))
    dump(out / 'final/submission.ipynb', new)
    meta = json.loads((SRC / 'kernel-metadata.json').read_text()); meta.pop('id_no', None)
    meta.update(id=kernel, title='11-12 v5.1 EB sibling experts', code_file='submission.ipynb', is_private=True)
    meta['dataset_sources'] = list(meta['dataset_sources']) + [dataset]
    dump(out / 'final/kernel-metadata.json', meta)
    dump(out / 'upload/dataset-metadata.json', dict(title='11-12 v5.1 accepted components', id=dataset,
         licenses=[dict(name='CC-BY-SA-4.0')]))
    dump(out / 'build.json', dict(decision=str(Path(decision_path).resolve()), decision_sha256=sha(decision_path),
         kernel=kernel, dataset=dataset, enabled=enabled, notebook_sha256=sha(out / 'final/submission.ipynb'), sha256=files_hash))
    return dict(kernel=kernel, dataset=dataset, out=str(out.resolve()), enabled=enabled)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, default=RUN / 'evidence/decision.json')
    p.add_argument('--out', type=Path, default=ROOT / 'submission_v51')
    p.add_argument('--username', required=True)
    a = p.parse_args(); print(json.dumps(build(a.decision, a.out, a.username), indent=2))
