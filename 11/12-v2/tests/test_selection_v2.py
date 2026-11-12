"""OOF reporting must deploy predeclared optimized states, including negative results."""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import shared
import public_parent as PP
from windows import sha
from selection import ba


def test():
    mock = types.ModuleType('common')
    y = np.tile(np.arange(18), 10)
    fold = np.repeat(np.arange(5), 36)
    public = np.eye(18)[y] * 6
    optimized = np.eye(18)[(y + 1) % 18] * 6
    mock.y_all = y
    mock.classes = [f'class{k:02}' for k in range(18)]
    mock.FOLDS = {k: np.flatnonzero(fold == k) for k in range(5)}
    mock.pid = np.repeat(np.arange(10), 18)
    mock.public_lora_stack = lambda: np.zeros_like(public)
    mock.public_native_oof = lambda: (public, public, fold)
    state = dict(plain_scale=1., centered_scale=1., selected_centered_weight=0.)
    mock.public_state = lambda k: state
    mock.public_bias = lambda k: np.zeros(18)
    mock.groups_sized = lambda prompts, *args: prompts
    mock.state_lp = lambda plain, *args: plain - np.logaddexp.reduce(plain, -1, keepdims=True)
    mock.ba = ba
    mock.lsm = lambda x: x - np.logaddexp.reduce(x, -1, keepdims=True)
    previous, argv = sys.modules.get('common'), sys.argv[:]
    original_parent = PP.load_parent
    parent_record = {'fixture': 'frozen sources'}
    PP.load_parent = lambda prompts, classes: (np.full((len(prompts), len(classes)), -np.log(len(classes))), parent_record)
    try:
        sys.modules['common'] = mock
        with tempfile.TemporaryDirectory(prefix='grpo_selection_') as directory:
            runs = Path(directory)
            for k, rows in mock.FOLDS.items():
                root = runs / f'fold{k}'; (root / 'calibration').mkdir(parents=True)
                np.savez(root / 'native_eval.npz', val=rows, plain_ours=optimized[rows],
                         centered_ours=optimized[rows], bias_ours=np.zeros(18))
                (root / 'trainables.safetensors').write_bytes(b'optimized fixture')
                (root / 'calibration/fold.json').write_text(json.dumps({'models': {'qwen8': state}}))
                done = dict(method='public_native_grpo_v2', voting={'parent_record': parent_record},
                            calibration_mode='frozen_public', optimizer_steps=2,
                            public_sha256={'state': 'different source weights'},
                            artifact_sha256={n: sha(root / n) for n in ['trainables.safetensors']})
                (root / 'done.json').write_text(json.dumps(done))
                check = dict(parity_ok=True, artifact_sha256={n: sha(root / n) for n in
                             ['done.json', 'trainables.safetensors', 'native_eval.npz', 'calibration/fold.json']})
                (root / 'native_eval.json').write_text(json.dumps(check))
            sys.argv = ['decide.py', '--runs', str(runs), '--out', str(runs / 'decision.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'decide.py'), run_name='__main__')
            result = json.loads((runs / 'decision.json').read_text())
            assert result['decision'] == 'grpo' and result['our_folds_used'] == list(range(5))
            assert result['candidates']['overall_vs_original']['delta'] < 0
            assert not result['candidates']['overall_vs_original']['positive_uplift_evidence']
            # A predeclared subset uses only those OOF rows, even if every fold
            # is present on disk. The choice is not made using observed uplift.
            sys.argv += ['--folds', '1,3']
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'decide.py'), run_name='__main__')
            result = json.loads((runs / 'decision.json').read_text())
            assert result['our_folds_used'] == [1, 3] and result['candidates']['overall_vs_original']['rows'] == 72
            (runs / 'fold1/calibration/fold.json').write_text('{}')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPTS / 'decide.py'), run_name='__main__')
            except ValueError as error:
                assert 'Changed evaluated artifact' in str(error)
            else:
                raise AssertionError('Changed calibration was accepted')
    finally:
        sys.argv = argv
        PP.load_parent = original_parent
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    print(json.dumps(dict(optimized_states_required=True, negative_results_reported=True,
                          declared_folds_only=True, changed_calibration_rejected=True)))


if __name__ == '__main__':
    test()
