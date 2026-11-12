"""The two-fold proxy cannot borrow models from undeployed folds."""
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
from windows import sha
from selection import ba


def test():
    mock = types.ModuleType('common')
    n = 36
    y = np.tile(np.arange(18), 8)
    folds = np.repeat(np.arange(4), n)
    plain_public, plain_ours = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for row in range(len(y)):
        if folds[row] < 2:
            plain_public[row, (y[row] + 1) % 18] = .1
            plain_ours[row, y[row]] = 6.
        else:
            plain_public[row, y[row]] = 6.
            plain_ours[row, (y[row] + 1) % 18] = 6.
    mock.y_all = y
    mock.FOLDS = {k: np.flatnonzero(folds == k) for k in range(4)}
    mock.pid = np.repeat(np.arange(8), 18)
    mock.pidx = mock.pid
    mock.public_lora_stack = lambda: np.zeros_like(plain_public)
    mock.public_native_oof = lambda: (plain_public, plain_public, folds)
    state = dict(plain_scale=1., centered_scale=1., selected_centered_weight=0.)
    mock.public_state = lambda k: state
    mock.public_bias = lambda k: np.zeros(18)
    mock.groups_sized = lambda prompts, *args: prompts
    mock.state_lp = lambda plain, *args: plain
    mock.ba = ba
    def lsm(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    mock.lsm = lsm
    original_common, original_argv = sys.modules.get('common'), sys.argv[:]
    try:
        sys.modules['common'] = mock
        with tempfile.TemporaryDirectory(prefix='fold_selection_') as directory:
            runs = Path(directory)
            for k, rows in mock.FOLDS.items():
                root = runs / f'fold{k}'; (root / 'calibration').mkdir(parents=True)
                np.savez(root / 'native_eval.npz', val=rows,
                         plain_ours=plain_ours[rows], centered_ours=plain_ours[rows], bias_ours=np.zeros(18))
                (root / 'calibration/fold.json').write_text(json.dumps({'models': {'qwen8': state}}))
                check = dict(parity_ok=True, artifact_sha256={name: sha(root / name)
                    for name in ['native_eval.npz', 'calibration/fold.json']})
                (root / 'native_eval.json').write_text(json.dumps(check))
            sys.argv = ['decide.py', '--runs', str(runs), '--out', str(runs / 'decision.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'decide.py'), run_name='__main__')
            result = json.loads((runs / 'decision.json').read_text())
            assert result['decision'] == 'ours2'
            assert result['our_folds_used'] == [0, 1]
            assert result['candidates']['ours2']['rows'] == 2 * n
            assert result['candidates']['ours4']['rows'] == 4 * n
            (runs / 'fold0/calibration/fold.json').write_text('{}')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPTS / 'decide.py'), run_name='__main__')
            except ValueError as error:
                assert 'Changed evaluated artifact' in str(error)
            else:
                raise AssertionError('Changed calibration was accepted')
    finally:
        sys.argv = original_argv
        if original_common is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = original_common
    print(json.dumps(dict(only_deployed_folds_used=True, changed_calibration_rejected=True)))


if __name__ == '__main__':
    test()
