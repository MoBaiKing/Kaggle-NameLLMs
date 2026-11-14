"""CPU: final SNR edit on all 2,699 clean OOF rows (lam by 5-fold prompt CV) and the per-fold head tensors.
Writes artifacts/snr_edit.json and upload/snr_fold{k}_heads.safetensors (the four rewritten head tensors of the
public fold-k state; every other tensor stays the public one and is merged on Kaggle after a SHA256 check)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared  # noqa: E402
import common as C  # noqa: E402
import neuron_rewrite as N  # noqa: E402
from oof_targets import sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEADS = ('plain.weight', 'plain.bias', 'centered.weight', 'centered.bias')


def head_tensors(edit, public_file):
    from safetensors.numpy import load_file
    t = load_file(str(public_file))
    out = {}
    for head in ('plain', 'centered'):
        D, d = edit[head]
        A = np.eye(18) + D
        out[f'{head}.weight'] = (A @ t[f'{head}.weight'].astype(np.float64)).astype(np.float32)
        out[f'{head}.bias'] = (A @ t[f'{head}.bias'].astype(np.float64) + d).astype(np.float32)
    return out


def main():
    from safetensors.numpy import save_file
    o = N.load_oof()
    rows = np.arange(len(o['y']))
    lam, scores = N.choose_lam(o, rows)
    edit = N.fit(o, rows, lam, 99)
    up = ROOT / 'upload'; up.mkdir(exist_ok=True)
    manifest = json.loads((C.NW / 'manifest.json').read_text())
    record = dict(method='v5_snr_sibling_neuron_rewrite', lam=lam, inner_cv_nll=scores, edit=N.to_json(edit),
                  mask='diagonal + families.NEIGHBORS', public_sha256={}, heads_sha256={},
                  in_sample_native_nll=dict(public=N.nll_rows(o, rows, None), snr=N.nll_rows(o, rows, edit)))
    for k in range(5):
        public = C.NW / f'fold{k}/trainables.safetensors'
        record['public_sha256'][str(k)] = sha(public)
        if record['public_sha256'][str(k)] != manifest['file_hashes'][f'fold{k}/trainables.safetensors']:
            raise SystemExit(f'Public fold {k} differs from the pinned manifest')
        target = up / f'snr_fold{k}_heads.safetensors'
        save_file(head_tensors(edit, public), str(target))
        record['heads_sha256'][str(k)] = sha(target)
    (ROOT / 'artifacts/snr_edit.json').write_text(json.dumps(record, indent=1) + '\n')
    spec = {k: record[k] for k in ('method', 'lam', 'public_sha256', 'heads_sha256')}
    (up / 'dyl_v5_snr.json').write_text(json.dumps(spec) + '\n')
    print(json.dumps(dict(lam=lam, inner_cv_nll={str(k): round(v, 5) for k, v in scores.items()},
                          in_sample_native_nll={k: round(v, 4) for k, v in record['in_sample_native_nll'].items()},
                          heads_sha256=record['heads_sha256']), indent=1))


if __name__ == '__main__':
    main()
