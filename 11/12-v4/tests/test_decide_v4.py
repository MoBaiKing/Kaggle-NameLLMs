"""End to end on the real pinned OOF (CPU, no model): synthetic A1 eval outputs (= public native) and synthetic A2
scores -> decide_v4 (anchored proxy, nested CV) -> decision + B bundle -> Notebook build. Informative A2 must raise
S3 over S1 with a positive interval; random A2 must not cause clear harm. The built Notebook differs from the
pinned one only in the setup cell, two new %%writefile cells and the three infer.py anchors; the setup patch runs
offline against mock Kaggle inputs and rejects a wrong A2 base."""
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
import decide_v4 as D
import build_submission as B
from oof_targets import sha


def a1_family(root):
    pp, pc, _ = C.public_native_oof()
    for k in range(5):
        run = root / f'fold{k}'; run.mkdir(parents=True)
        val = C.FOLDS[k]
        np.savez(run / 'native_eval.npz', val=val, plain_wise050=pp[val], centered_wise050=pc[val], bias_wise050=C.public_bias(k))
        save_file({'plain.weight': torch.full((18, 4), float(k + 1))}, run / 'trainables_wise050.safetensors')
        rec = dict(parity_ok=True, artifact_sha256={n: sha(run / n) for n in ['native_eval.npz', 'trainables_wise050.safetensors']})
        (run / 'native_eval.json').write_text(json.dumps(rec))


def a2_family(root, informative, seed):
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    S = -150 + rng.normal(size=(C.N, 18)) * 3
    if informative:
        hit = rng.uniform(size=C.N) < .6
        S[np.flatnonzero(hit), C.y_all[hit]] += 25
    np.savez(root / 'a2_scores.npz', official_sums=S, official_tokens=np.full(C.N, 120.), official_done=np.ones(C.N, bool))
    (root / 'adapters.safetensors').write_bytes(b'adapters' + bytes([seed]))
    bundle = dict(adapters_sha256=sha(root / 'adapters.safetensors'), base={'config.json': 'cfg'})
    (root / 'a2_bundle.json').write_text(json.dumps(bundle))
    (root / 'a2_scores.json').write_text(json.dumps(dict(scores_sha256=sha(root / 'a2_scores.npz'), adapters_sha256=bundle['adapters_sha256'],
                                                         official_ba_alone=0., dev_ba_alone=0.)))


def decide(base, a2):
    argv = sys.argv[:]
    sys.argv = ['decide_v4.py', '--a1', str(base / 'a1'), '--a2', str(a2), '--out', str(base / ('b_' + a2.name))]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            D.main()
    finally:
        sys.argv = argv
    return json.loads((base / ('b_' + a2.name) / 'decision.json').read_text())


def notebook_checks(base, plan):
    nb = json.loads(Path(plan['notebook']).read_text())
    pinned = json.loads((B.SRC / 'ntl-684-ozertuu-mirror.ipynb').read_text())
    engine = next(i for i, c in enumerate(pinned['cells']) if ''.join(c['source']).startswith('%%writefile infer.py'))
    assert len(nb['cells']) == len(pinned['cells']) + 3
    new = nb['cells'][1:]
    assert ''.join(new[engine]['source']).startswith('%%writefile dyl_ccl.py') and ''.join(new[engine + 1]['source']).startswith('%%writefile dyl_sage.py')
    rest = new[:engine] + new[engine + 2:]
    diff = [i for i, (a, b) in enumerate(zip(pinned['cells'], rest)) if a['source'] != b['source']]
    assert diff == [2, engine], diff
    before, after = ''.join(pinned['cells'][engine]['source']), ''.join(rest[engine]['source'])
    assert B.edit_infer(before) == after
    compile(after.partition('\n')[2], 'infer.py', 'exec')
    for i in (engine, engine + 1):
        compile(''.join(new[i]['source']).partition('\n')[2], 'dyl', 'exec')
    assert ''.join(new[engine + 1]['source']).partition('\n')[2] == (Path(B.__file__).with_name('sage.py')).read_text()
    meta = json.loads((base / 'submission/final/kernel-metadata.json').read_text())
    assert 'qwen-lm/qwen-3/Transformers/1.7b-base/1' in meta['model_sources'] and meta['is_private']
    # run the setup patch offline
    inp = base / 'input'; public = base / 'public'
    (public / 'base').mkdir(parents=True); (public / 'calibration').mkdir(); (public / 'manifest.json').write_text('{}')
    for k in range(5):
        (public / f'fold{k}').mkdir()
        save_file({'plain.weight': torch.zeros(18, 4)}, public / f'fold{k}/trainables.safetensors')
        (public / f'calibration/fold{k}.json').write_text(json.dumps({'fold': k}))
    shutil.copytree(base / 'submission/upload', inp / 'ours')
    model = inp / 'models/qwen/1.7b-base/1'; model.mkdir(parents=True)
    (model / 'model.safetensors').write_bytes(b'w' * 1000); (model / 'config.json').write_text('{"tiny": 1}')
    spec = json.loads((inp / 'ours' / B.SPEC_NAME).read_text())
    spec.update(a2_base_bytes=1000, a2_base_sha256=sha(model / 'model.safetensors'), a2_base_config_sha256=sha(model / 'config.json'))
    (inp / 'ours' / B.SPEC_NAME).write_text(json.dumps(spec))
    code = B.PATCH.replace('/kaggle/input', str(inp)).replace('/kaggle/temp/dyl_native_v4_mix', str(base / 'mix'))
    scope = dict(NATIVE=str(public), glob=glob, os=os, json=json, shutil=shutil)
    exec(compile(code, 'setup', 'exec'), scope)
    assert os.environ['DYL_V4_A2_BASE'] == str(model) and os.path.exists(os.environ['DYL_V4_B_BUNDLE'])
    spec['a2_base_sha256'] = '0' * 64; (inp / 'ours' / B.SPEC_NAME).write_text(json.dumps(spec))
    try:
        exec(compile(B.PATCH.replace('/kaggle/input', str(inp)).replace('/kaggle/temp/dyl_native_v4_mix', str(base / 'mix2')), 'setup', 'exec'),
             dict(NATIVE=str(public), glob=glob, os=os, json=json, shutil=shutil))
    except AssertionError:
        pass
    else:
        raise AssertionError('A wrong A2 base was accepted')


def test(base):
    a1_family(base / 'a1')
    a2_family(base / 'informative', True, 1); a2_family(base / 'random', False, 2)
    good = decide(base, base / 'informative'); noise = decide(base, base / 'random')
    g, n = good['evidence']['S3_full_vs_S1'], noise['evidence']['S3_full_vs_S1']
    assert good['decision'] == 'v4_full' and good['public_fallback'] is False
    assert g['delta'] > 3 and g['ci95'][0] > 0, g
    assert n['ci95'][1] > -.5 and n['delta'] > -.6, n
    assert abs(good['evidence']['S1_A1_vs_S0']['delta']) < 1e-9
    plan = B.build(base / 'b_informative/decision.json', base / 'submission', 'reproducer')
    notebook_checks(base, plan)
    print(json.dumps(dict(informative_a2=dict(delta=round(g['delta'], 2), ci95=[round(x, 2) for x in g['ci95']]),
                          random_a2=dict(delta=round(n['delta'], 2), ci95=[round(x, 2) for x in n['ci95']]),
                          notebook_only_expected_changes=True, setup_patch_offline=True, wrong_a2_base_rejected=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v4_decide_') as d:
        test(Path(d))
