"""Shared cell sampling and losses; preserved from the original training implementation."""
import math,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

LAB = {'claude': 0, 'gemini': 1, 'gpt': 2}

def supcon(z, labels, t):
    """Supervised contrastive loss; positives share the label, same-prompt rows of other models are hard negatives."""
    sim = z @ z.T / t
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None]) & ~eye
    logprob = sim.masked_fill(eye, -1e9).log_softmax(1)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    return -(logprob * pos).sum(1)[has].div(pos.sum(1)[has]).mean()

def official_cells(prompt_ids, labels, rows, ppb, mpb, max_cell, rng):
    """Original cells on official rows: prompt-group x model-group, each row once per official epoch, split above max_cell."""
    prompts = rng.permutation(np.unique(prompt_ids[rows])); models = rng.permutation(np.unique(labels[rows]))
    pg = [prompts[i:i + ppb] for i in range(0, len(prompts), ppb)]; mg = [models[i:i + mpb] for i in range(0, len(models), mpb)]
    by = {}
    for r in rows:
        by.setdefault((prompt_ids[r], labels[r]), []).append(int(r))
    out = []
    for g in pg:
        for m in mg:
            cell = [r for p in g for c in m for r in by.get((p, c), [])]
            if not cell:
                continue
            if len(cell) > max_cell:
                cell = [cell[i] for i in rng.permutation(len(cell))]; k = math.ceil(len(cell) / max_cell)
                out.extend([cell[i::k] for i in range(k)])
            else:
                out.append(cell)
    return [out[i] for i in rng.permutation(len(out))]

def pool_cells(prompt_ids, labels, rows, lab_of, rng, bs=16, k=4, p_family=0.5, scan=24):
    """Real-prompt cells on sparse data. Each batch: k focus models (one lab with prob p_family, else any; both drawn in
    proportion to the rows still unused), then prompts carrying the most focus models are added, one row per
    (prompt, focus model), until bs rows; a short batch is topped up with other models of the same prompts.
    Every row is used exactly once; leftovers below bs/2 are merged. Batches are shuffled at the end."""
    rows = [int(r) for r in rng.permutation(np.asarray(rows))]
    prompt_ids, labels = {r: str(prompt_ids[r]) for r in rows}, {r: int(labels[r]) for r in rows}
    by, by_prompt, cls_rows = {}, {}, {}
    for r in rows:
        by.setdefault((prompt_ids[r], labels[r]), []).append(r); by_prompt.setdefault(prompt_ids[r], set()).add(labels[r])
        cls_rows.setdefault(labels[r], []).append(r)
    used = np.zeros(max(rows) + 1, bool); left = {c: len(v) for c, v in cls_rows.items()}

    def take(pr, c):
        lst = by.get((pr, c))
        while lst:
            r = lst.pop()
            if not used[r]:
                used[r] = True; left[c] -= 1
                return r
        return None

    def tail(c):                                                   # prompts of up to `scan` unused rows of class c
        lst = cls_rows[c]
        while lst and used[lst[-1]]:
            lst.pop()
        out, i = [], len(lst) - 1
        while i >= 0 and len(out) < scan:
            if not used[lst[i]]:
                out.append(prompt_ids[lst[i]])
            i -= 1
        if len(lst) - 1 - i > 4 * scan:                            # tail mostly used: compact once, keeps the scan O(scan)
            lst[:] = [r for r in lst if not used[r]]
        return out

    out, small = [], []
    while True:
        live = [c for c, n in left.items() if n > 0]
        if not live:
            break
        wts = np.array([left[c] for c in live], float)
        if rng.random() < p_family:
            labs = sorted({lab_of[c] for c in live}); lw = np.array([sum(left[c] for c in live if lab_of[c] == f) for f in labs], float)
            f = labs[rng.choice(len(labs), p=lw / lw.sum())]
            cand = [c for c in live if lab_of[c] == f]; cw = np.array([left[c] for c in cand], float)
            focus = list(rng.choice(cand, size=min(k, len(cand)), replace=False, p=cw / cw.sum()))
        else:
            focus = []
        rest = [c for c in live if c not in focus]
        if len(focus) < k and rest:
            rw = np.array([left[c] for c in rest], float)
            focus += list(rng.choice(rest, size=min(k - len(focus), len(rest)), replace=False, p=rw / rw.sum()))
        focus = [int(c) for c in focus]
        cands = list(dict.fromkeys(pr for c in focus for pr in tail(c)))
        score = [sum(1 for c in focus if any(not used[r] for r in by.get((pr, c), ()))) for pr in cands]
        order = sorted(range(len(cands)), key=lambda i: (-score[i], rng.random()))
        batch, chosen = [], []
        for i in order:
            if len(batch) >= bs:
                break
            chosen.append(cands[i])
            for c in focus:
                if len(batch) < bs:
                    r = take(cands[i], c)
                    if r is not None:
                        batch.append(r)
        if len(batch) < bs:                                        # top up: other models answering the same prompts
            for pr in chosen:
                for c in sorted(by_prompt[pr] - set(focus)):
                    if len(batch) < bs:
                        r = take(pr, c)
                        if r is not None:
                            batch.append(r)
        assert batch, 'no progress'
        (out if len(batch) >= bs // 2 else small).append(batch)
    flat = [r for b in small for r in b]
    out += [flat[i:i + bs] for i in range(0, len(flat), bs)]
    return [out[i] for i in rng.permutation(len(out))]

def lr_factor(p, warmup):
    """Wall-clock schedule: linear warm-up on [0, warmup), cosine to 0 on [warmup, 1]."""
    if p < warmup:
        return max(p / warmup, 0.02)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (p - warmup) / (1.0 - warmup))))

def atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)
