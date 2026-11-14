"""Package three public model screening outputs (e.g. native8B, Qwen14B, Llama8B).
Each input NPZ: ids[N], labels[18], probs[N,18]; JSON sidecar: payload_sha256,
source_sha256, model_id, checkpoint_sha256. Scores must come from the unverified file, never training labels.
This step consumes saved GPU inference only; it neither calls APIs nor invents pseudo evidence.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v51_common import ROOT, CLASSES, sha, load_verified, save_npz, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, nargs=3, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'cache/gemini37_screen.npz')
    a = p.parse_args()
    source = ROOT / 'cache/open_pool_latest/gemini37_unverified.parquet'
    frame = pd.read_parquet(source); outputs, records = [], []
    for path in a.scores:
        rec = load_verified(path)
        if rec['source_sha256'] != sha(source) or not rec['checkpoint_sha256']:
            raise ValueError('Missing model/source provenance for screening')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['ids'].astype(str), frame.id.astype(str)) or z['labels'].tolist() != CLASSES:
                raise ValueError('Screen score row/class order changed')
            outputs.append(z['probs'])
        records.append(rec)
    if len({r['model_id'] for r in records}) != 3 or len({r['checkpoint_sha256'] for r in records}) != 3:
        raise ValueError('Three folds/aliases of one checkpoint are not independent screening models')
    save_npz(a.out, probs=np.stack(outputs), ids=frame.id.astype(str).values, labels=np.array(CLASSES))
    dump(a.out.with_suffix('.json'), dict(payload_sha256=sha(a.out), source_sha256=sha(source),
         model_ids=[r['model_id'] for r in records], models=records))


if __name__ == '__main__':
    main()
