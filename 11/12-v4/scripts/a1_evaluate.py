"""GPU inference only: the unchanged v3 native evaluation (public native_q8 runtime; slots alpha 1 / public /
WiSE alpha 0.5; public-runtime and export parity) applied to an A1 run directory."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared  # noqa: F401,E402
import evaluate_v3 as E  # noqa: E402
from oof_targets import sha  # noqa: E402


def identity(run, k):
    done = json.loads((run / 'done.json').read_text())
    if done['args']['fold'] != k or done['method'] != 'v4_a1_hsc_native':
        raise ValueError('Run is not the requested completed A1 fold')
    for name in ('trainables.safetensors', 'trainables_wise050.safetensors', 'val_probs_final.npy', 'val_rows.npy'):
        if sha(run / name) != done['artifact_sha256'][name]:
            raise ValueError(f'Changed A1 artifact: {name}')
    return dict(family='v4_a1', done_sha256=sha(run / 'done.json'))


if __name__ == '__main__':
    E.identity = identity
    E.main()
