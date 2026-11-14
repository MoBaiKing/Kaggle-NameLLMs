"""Kaggle runtime: EB consumes retained E_q; SAGE scores each routed row with ONE 8B expert."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from v51_common import CLASSES, sha, dump
import v51_eb as EB
import v51_expert as E
import v51_sage as B


def bundle():
    folder = Path(os.environ['DYL_V51_BUNDLE'])
    spec = json.loads((folder / 'dyl_v51_bundle.json').read_text())
    if spec['method'] != 'v51_eb_sibling_experts' or spec['classes'] != CLASSES:
        raise ValueError('Unexpected v5.1 bundle')
    for name, expected in spec['sha256'].items():
        if sha(folder / name) != expected:
            raise ValueError(f'Changed deployment artifact: {name}')
    return folder, spec


def native(plain, centered, bias, topic, groups, states, labels):
    folder, spec = bundle()
    if labels != CLASSES or topic is None or len(topic) != plain.shape[1] or not np.isfinite(topic).all():
        raise ValueError('Missing/misaligned existing Qwen topic vectors')
    with np.load(folder / 'eb_models.npz', allow_pickle=False) as z:
        models, taus = EB.unpack(z)
    return EB.calibrate(plain, centered, bias, topic, groups, states, models, taus)


def rerank(probs, test, query, log, write_pred, labels, base, work, deadline):
    import sage as S
    import torch
    folder, spec = bundle(); enabled = spec['enabled']['experts']
    if labels != CLASSES:
        raise ValueError('Runtime class order changed')
    if not enabled:
        return probs.argmax(1)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    routed = E.route(probs, enabled); rows = np.flatnonzero(routed >= 0)
    if not len(rows):
        log('v5.1 no rows routed to experts'); return probs.argmax(1)
    jobs = []; lp = np.full_like(probs, np.nan)
    try:
        if torch.cuda.device_count() < 2:
            raise ValueError('Two Kaggle GPUs required for FP16 8B expert scoring')
        # One sharded worker owns BOTH T4s, like native; independent 8B workers would OOM 16GB cards.
        for gpu in range(1):
            own = rows
            if not len(own):
                continue
            req = work / f'request{gpu}.json'; out = work / f'scores{gpu}.npz'
            if out.exists():
                out.unlink()
            dump(req, dict(query=query, base=base, bundle=str(folder), enabled=enabled,
                           rows=own.tolist(), route=routed.tolist(), out=str(out), hashes=spec['sha256']))
            proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('v51_expert.py')), str(req)],
                                    env=dict(os.environ))
            jobs.append((proc, out, own))
        for proc, out, own in jobs:
            proc.wait(timeout=max(1., deadline - time.time()))
            if proc.returncode or not out.exists():
                raise RuntimeError('Expert scoring failed')
            with np.load(out, allow_pickle=False) as z:
                if not np.array_equal(z['rows'], own) or z['lp'].shape != probs.shape:
                    raise ValueError('Expert output alignment changed')
                lp[own] = z['lp'][own]
    finally:
        for proc, _, _ in jobs:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    b = json.loads((folder / 'sage_bundle.json').read_text())
    pred = B.apply(probs, lp, enabled, b['weights'], S)
    log('v5.1 expert rows', len(rows), 'changed', int((pred != probs.argmax(1)).sum()))
    write_pred(pred, 'v5.1 accepted discriminative experts + SAGE')
    return pred
