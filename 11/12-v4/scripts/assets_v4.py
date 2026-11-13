"""Download and verify the A2 base: Qwen/Qwen3-1.7B-Base at a pinned revision. The Kaggle model
qwen-lm/qwen-3/transformers/1.7b-base/1 ships the same model.safetensors (3,441,185,608 bytes); the notebook
re-hashes it against MODEL_SHA256 before scoring."""
import json
from pathlib import Path

from shared import ROOT
from oof_targets import sha

REPO = 'Qwen/Qwen3-1.7B-Base'
REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
MODEL_SHA256 = '6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
MODEL_BYTES = 3441185608
KAGGLE_MODEL = 'qwen-lm/qwen-3/Transformers/1.7b-base/1'
DEFAULT = ROOT / 'external/qwen3-1.7b-base'


def fetch(target=DEFAULT):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=REVISION, local_dir=target,
                      allow_patterns=['*.json', '*.safetensors', '*.txt'])
    return verify(target)


def verify(target=DEFAULT):
    model = Path(target) / 'model.safetensors'
    if model.stat().st_size != MODEL_BYTES or sha(model) != MODEL_SHA256:
        raise SystemExit('A2 base weights differ from the pinned revision')
    out = dict(repo=REPO, revision=REVISION, model_sha256=MODEL_SHA256, path=str(target),
               config_sha256=sha(Path(target) / 'config.json'))
    print(json.dumps(out))
    return out


if __name__ == '__main__':
    import sys
    (verify if '--verify' in sys.argv else fetch)()
