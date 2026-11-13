"""GPU inference only: CCL-Gen scores of every class for all 2,699 official rows (clean: A2 never trained on
official rows or their prompt topics) and the 1,080 external dev rows. fp16, like the Kaggle T4 runtime.
Run with torchrun --nproc_per_node=4; rank r scores rows r::4 and rank 0 merges."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from shared import BASE, ROOT
from oof_targets import sha
import ccl_gen as G
import external_v4 as E
import families as F


def main(argv=None, factory=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--run', type=Path, default=ROOT / 'runs_v4/a2')
    p.add_argument('--external', type=Path, default=E.DEFAULT)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if a.device == 'cuda' else 'cpu')
    if world > 1:
        dist.init_process_group('nccl' if a.device == 'cuda' else 'gloo')
    try:
        done = json.loads((a.run / 'done.json').read_text())
        if sha(a.run / 'adapters.safetensors') != done['adapters_sha256'] or sha(a.run / 'a2_bundle.json') != done['bundle_sha256']:
            raise ValueError('A2 artifacts changed after training')
        dtype = torch.float16 if a.device == 'cuda' else torch.float32
        if factory is None:
            meta, tok, model = G.load_bundle(str(a.run), str(a.base), device, dtype)
        else:
            from safetensors.torch import load_file
            meta = json.loads((a.run / 'a2_bundle.json').read_text())
            tok, model = factory(str(a.base), len(F.CLASSES), device, dtype, None)
            model.load_trainable(load_file(str(a.run / 'adapters.safetensors'))); model.eval()
        official = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
        external = pd.read_parquet(a.external)
        dev = external[external.a2_dev].reset_index(drop=True)
        sets = {'official': official.text.fillna('').astype(str).values, 'dev': dev.text.astype(str).values}
        if a.smoke:
            sets = {k: v[:a.smoke] for k, v in sets.items()}
        allc = np.tile(np.arange(len(F.CLASSES)), (1, 1))
        for name, texts in sets.items():
            rows = np.arange(rank, len(texts), world)
            enc = G.encode(tok, texts[rows], meta['max_tokens'])
            cand = np.repeat(allc, len(rows), 0)
            sums, tokens, ok = G.score(model, tok, enc, cand, device, max(1, a.batch // 4))
            np.savez(a.run / f'.score_{name}_{rank}.npz', rows=rows, sums=sums, tokens=tokens, done=ok)
        if world > 1:
            dist.barrier()
        if rank == 0:
            out = {}
            for name, texts in sets.items():
                S = np.zeros((len(texts), len(F.CLASSES))); N = np.zeros(len(texts)); D = np.zeros(len(texts), bool)
                for r in range(world):
                    with np.load(a.run / f'.score_{name}_{r}.npz') as z:
                        S[z['rows']] = z['sums']; N[z['rows']] = z['tokens']; D[z['rows']] = z['done']
                    (a.run / f'.score_{name}_{r}.npz').unlink()
                out[name] = (S, N, D)
            y_off = np.searchsorted(F.CLASSES, official.model.values[:len(out['official'][0])])
            y_dev = dev.label.values[:len(out['dev'][0])]
            target = a.run / ('a2_scores_smoke.npz' if a.smoke else 'a2_scores.npz')
            np.savez(target, official_sums=out['official'][0], official_tokens=out['official'][1], official_done=out['official'][2],
                     dev_sums=out['dev'][0], dev_tokens=out['dev'][1], dev_done=out['dev'][2], classes=np.array(F.CLASSES))
            ba = lambda y, p: float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))
            rec = dict(official_rows=int(len(y_off)), official_scored=int(out['official'][2].sum()),
                       official_ba_alone=round(100 * ba(y_off, out['official'][0].argmax(1)), 2),
                       dev_ba_alone=round(100 * ba(y_dev, out['dev'][0].argmax(1)), 2),
                       scores_sha256=sha(target), adapters_sha256=done['adapters_sha256'],
                       split='official rows are clean held-out (VALIDATION); dev = external prompt clusters (monitoring)')
            (a.run / ('a2_scores_smoke.json' if a.smoke else 'a2_scores.json')).write_text(json.dumps(rec, indent=1) + '\n')
            print(json.dumps(rec), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
