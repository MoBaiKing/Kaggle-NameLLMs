"""StyleRank 11/11: train in the public native Qwen3-Embedding-8B layout.
The unchanged vendor/native_q8.py (SHA256 in config/assets.json) runs exported fold states on Kaggle.

Layout (identical to the public expert, read from native_q8.py):
  * backbone Qwen3-Embedding-8B, all 36 layers; input = INSTRUCTION + text, first 511 tokens + <|endoftext|> (151643), right pad
  * last-token states at depths 3,6,9,12,18,24,30,36 (36 = last_hidden_state, L2-normalised), fp32 LayerNorm per depth,
    concatenated (8 x 4096) -> two fp32 linear heads: plain and centred (centred = logits minus their prompt-group mean + bias)
  * LoRA r16, scale 2.0 (alpha 32) on q,k,v,o,gate,up,down of every layer, A/B fp32 -> exported with the public key names
Our method (shared, unchanged where it applies):
  * logit-adjusted (Menon et al. 2021) + importance-weighted CE, top-5 hard CE, SupCon on a projection (training only),
    feature dropout, real-prompt pool cells (4 focus models, same lab p 0.5), exact GradCache, wall-clock cosine, resume
  * new: the centred head is trained on every multi-answer prompt group inside a cell (pool cells too; the public expert
    trained it on official rows only), coefficient 0.5
  * new: stage 2 = official-train-only finish (--stage2-epochs over official cells, cosine from --lr2), after the pool pass
  * an inner calibration set (--inner-prompts prompts of the training fold) is held out of BOTH stages; native_eval.py
    fits the public calibration fields (plain/centred scale, centred weight) on it. The outer fold is never used to choose.
Outputs in <out>/fold<k>/: trainables.safetensors (public keys), done.json, val_probs_*.npy (monitoring only)."""
import argparse, json, math, os, signal, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import LAB, supcon, official_cells, pool_cells, lr_factor, atomic_save   # noqa: E402  (shared code, unchanged)

INSTRUCTION = 'Instruct: Identify the language model that generated the text based on its writing style.\nQuery: '
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
RANK, SCALE, TOKEN_LIMIT, TERMINAL = 16, 2.0, 512, 151643


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--model', required=True)
    p.add_argument('--folds', required=True); p.add_argument('--fold', type=int, required=True); p.add_argument('--out', required=True)
    p.add_argument('--stream', required=True)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--lr2', type=float, default=5e-5)
    p.add_argument('--proj-dim', type=int, default=256); p.add_argument('--lambda-con', type=float, default=0.2)
    p.add_argument('--temperature', type=float, default=0.1); p.add_argument('--lambda-hard', type=float, default=0.5)
    p.add_argument('--hard-k', type=int, default=5); p.add_argument('--lambda-cent', type=float, default=0.5)
    p.add_argument('--feat-drop', type=float, default=0.1); p.add_argument('--lora-drop', type=float, default=0.05)
    p.add_argument('--logit-adjust', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=16); p.add_argument('--focus-models', type=int, default=4)
    p.add_argument('--p-family', type=float, default=0.5); p.add_argument('--official-every', type=int, default=10)
    p.add_argument('--inner-prompts', type=int, default=18); p.add_argument('--stage2-epochs', type=int, default=2)
    p.add_argument('--train-end', required=True, help='ISO time: pool-stage cosine reaches 0 here')
    p.add_argument('--stage2-end', required=True, help='ISO time: save and pause incomplete stage 2 at this deadline')
    p.add_argument('--warmup', type=float, default=0.03); p.add_argument('--ckpt-min', type=float, default=20.0)
    p.add_argument('--max-steps', type=int, default=0, help='code tests: pool-stage steps'); p.add_argument('--max-steps2', type=int, default=0)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda')
    p.add_argument('--mem-cap-gb', type=float, default=28.5); p.add_argument('--micro-batch', type=int, default=16)
    p.add_argument('--monitor', action='store_true', help='Optional intermediate validation; disabled for speed')
    return p.parse_args(argv)


class TaskLoRA(nn.Module):
    """public native_q8.TaskLoRA + dropout on the LoRA input during training; A/B in fp32 as on Kaggle."""

    def __init__(self, layer, drop):
        super().__init__()
        self.base_layer = layer; dev = layer.weight.device
        self.a = nn.Linear(layer.in_features, RANK, bias=False, device=dev, dtype=torch.float32)
        self.b = nn.Linear(RANK, layer.out_features, bias=False, device=dev, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        result = self.base_layer(x)
        return (result + self.b(self.a(self.drop(x).float())) * SCALE).to(result.dtype)


class NativeStyle(nn.Module):
    def __init__(self, backbone, hidden, n_classes, proj_dim, feat_drop):
        super().__init__()
        self.backbone = backbone; self.n_layers = backbone.config.num_hidden_layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in DEPTHS])
        width = hidden * len(DEPTHS)
        self.plain = nn.Linear(width, n_classes); self.centered = nn.Linear(width, n_classes)
        self.drop = nn.Dropout(feat_drop)
        self.proj = nn.Sequential(nn.Linear(width, hidden // 4), nn.GELU(), nn.Linear(hidden // 4, proj_dim))

    def features(self, input_ids, attention_mask):
        """= native_q8.NativeQ8._features (same states, same normalisation)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
        assert len(out.hidden_states) == self.n_layers + 1
        pos = attention_mask.sum(1) - 1; rows = torch.arange(len(input_ids), device=input_ids.device)
        parts = []
        for depth, norm in zip(DEPTHS, self.norms):
            hs = out.last_hidden_state if depth == self.n_layers else out.hidden_states[depth]
            pooled = hs[rows, pos].float()
            if depth == self.n_layers:
                pooled = F.normalize(pooled, dim=-1)
            parts.append(norm(pooled))
        return torch.cat(parts, -1)

    def forward(self, input_ids, attention_mask):
        f = self.features(input_ids, attention_mask); fd = self.drop(f)
        return self.plain(fd), self.centered(fd), F.normalize(self.proj(f), dim=-1)


def encode_first_window(tok, texts):
    """first window of native_q8.windows(): tokens of INSTRUCTION + text, first 511, then the terminal token."""
    full = tok([INSTRUCTION + str(t) for t in texts], truncation=False, add_special_tokens=False)['input_ids']
    return [x[:TOKEN_LIMIT - 1] + [TERMINAL] for x in full]


def centred_loss(cent, bias, lab, w, groups, log_prior):
    """CE of (centred logits - prompt-group mean + bias), on rows whose group in this cell has >= 2 answers."""
    _, inv, cnt = torch.unique(groups, return_inverse=True, return_counts=True)
    elig = cnt[inv] >= 2
    if not elig.any():
        return cent.sum() * 0
    means = torch.zeros(len(cnt), cent.shape[1], device=cent.device).index_add_(0, inv, cent) / cnt[:, None]
    rel = cent - means[inv] + bias + log_prior
    return (F.cross_entropy(rel[elig], lab[elig], reduction='none') * w[elig]).sum() / w.sum().clamp_min(1e-6)


def cell_loss(plain, cent, z, lab, w, groups, bias, log_prior, args):
    lg = plain.float() + log_prior
    ce = (F.cross_entropy(lg, lab, reduction='none') * w).mean()
    with torch.no_grad():
        keep = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, lg.topk(args.hard_k, 1).indices, True)
        keep[torch.arange(len(lab)), lab] = True
    hard = (F.cross_entropy(lg.masked_fill(~keep, -1e4), lab, reduction='none') * w).mean()
    con = supcon(z.float(), lab, args.temperature)
    cl = centred_loss(cent.float(), bias, lab, w, groups, log_prior)
    return ce + args.lambda_hard * hard + args.lambda_con * con + args.lambda_cent * cl, ce, hard, con, cl


def train_step(model, tok, ids, y, wts, gid, cell, device, args, micro, log_prior):
    """cell loss backward; above `micro` rows exact GradCache (earlier/shared), now with three cached outputs."""
    lab = torch.tensor(y[cell], device=device); w = torch.tensor(wts[cell], device=device, dtype=torch.float32)
    groups = torch.tensor(gid[cell], device=device)
    enc = lambda rows: tok.pad({'input_ids': [ids[i] for i in rows]}, padding=True, return_tensors='pt').to(device)
    fwd = lambda e: model(e['input_ids'], e['attention_mask'])
    bias = model.centered.bias
    if len(cell) <= micro:
        pl, cn, z = fwd(enc(cell))
        loss, *parts = cell_loss(pl, cn, z, lab, w, groups, bias, log_prior, args); loss.backward()
        return [float(x) for x in parts]
    chunks = [cell[i:i + micro] for i in range(0, len(cell), micro)]; states, P, C, Z = [], [], [], []
    with torch.no_grad():
        for c in chunks:
            states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
            pl, cn, z = fwd(enc(c)); P.append(pl.float()); C.append(cn.float()); Z.append(z.float())
    Pc, Cc, Zc = (torch.cat(x).requires_grad_() for x in (P, C, Z))
    loss, *parts = cell_loss(Pc, Cc, Zc, lab, w, groups, bias, log_prior, args); loss.backward()
    sizes = [len(c) for c in chunks]
    for c, (cpu_st, cuda_st), gp, gc, gz in zip(chunks, states, Pc.grad.split(sizes), Cc.grad.split(sizes), Zc.grad.split(sizes)):
        torch.set_rng_state(cpu_st)
        if cuda_st is not None:
            torch.cuda.set_rng_state(cuda_st, device)
        pl, cn, z = fwd(enc(c))
        torch.autograd.backward([pl.float(), cn.float(), z.float()], [gp, gc, gz])
    return [float(x) for x in parts]


@torch.no_grad()
def predict(model, tok, ids, device, batch=8):
    """plain-head probabilities on the first window (monitoring only; the Kaggle path is native_eval.py)."""
    model.eval(); out = np.zeros((len(ids), model.plain.out_features), np.float32)
    for i, idx in enumerate([np.argsort([len(x) for x in ids])[s:s + batch] for s in range(0, len(ids), batch)]):
        e = tok.pad({'input_ids': [ids[j] for j in idx]}, padding=True, return_tensors='pt').to(device)
        out[idx] = model.plain(model.features(e['input_ids'], e['attention_mask'])).float().softmax(-1).cpu().numpy()
    model.train()
    return out


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if '.a.weight' in k or '.b.weight' in k or k.startswith(('plain.', 'centered.', 'proj.', 'norms.'))}


def export_public(model, path):
    """state in the public key layout (native_q8.NativeQ8.expected): fp32, exactly 2*252 LoRA + 2*8 norm + 4 head tensors."""
    from safetensors.torch import save_file
    out = {}
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, TaskLoRA):
            out[f'encoder.backbone.base_model.model.{name}.lora_A.default.weight'] = mod.a.weight.detach().float().cpu().contiguous()
            out[f'encoder.backbone.base_model.model.{name}.lora_B.default.weight'] = mod.b.weight.detach().float().cpu().contiguous()
    for prefix, m in (('norms', model.norms), ('plain', model.plain), ('centered', model.centered)):
        out.update({f'{prefix}.{n}': p.detach().float().cpu().contiguous() for n, p in m.named_parameters()})
    n_lora = model.n_layers * len(SUFFIXES)
    assert len(out) == 2 * n_lora + 2 * len(DEPTHS) + 4, len(out)
    tmp = str(path) + '.tmp'; save_file(out, tmp); os.replace(tmp, path)


def assemble(backbone, tok, args, n_classes, device):
    """LoRA-wrap a loaded Qwen3 backbone exactly where native_q8 does (every Linear with a public suffix) and add the heads."""
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert tok('', add_special_tokens=True)['input_ids'] == [TERMINAL], 'unexpected end-of-text handling'
    backbone.config.use_cache = False; backbone.requires_grad_(False)
    n = 0
    for name, mod in list(backbone.named_modules()):
        if isinstance(mod, nn.Linear) and name.rsplit('.', 1)[-1] in SUFFIXES:
            parent, attr = name.rsplit('.', 1); setattr(backbone.get_submodule(parent), attr, TaskLoRA(mod, args.lora_drop)); n += 1
    assert n == backbone.config.num_hidden_layers * len(SUFFIXES), n
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = NativeStyle(backbone, backbone.config.hidden_size, n_classes, args.proj_dim, args.feat_drop).to(device)
    for m in (model.norms, model.plain, model.centered, model.proj):
        m.float()
    for mod in backbone.modules():
        if isinstance(mod, TaskLoRA):
            mod.a.to(device=device, dtype=torch.float32).requires_grad_(True); mod.b.to(device=device, dtype=torch.float32).requires_grad_(True)
    return tok, model


def build_model(args, n_classes, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    backbone = AutoModel.from_pretrained(args.model, torch_dtype=dtype, attn_implementation='sdpa')
    assert backbone.config.num_hidden_layers == 36
    return assemble(backbone, tok, args, n_classes, device)


def inner_split(prompts_tr, n_inner, seed):
    """inner calibration prompts: deterministic choice among the training-fold prompts (never the outer fold)."""
    u = np.array(sorted(set(prompts_tr)))
    return set(np.random.default_rng([seed, 11]).choice(u, size=min(n_inner, len(u) // 4), replace=False).tolist())


def _run(args, rt, model_factory):
    from distributed import train_step as parallel_step, predict as parallel_predict
    from sklearn.metrics import balanced_accuracy_score
    device = rt.device
    log = lambda *a, **kw: print(*a, **kw) if rt.primary else None
    if device.type == 'cuda':
        total = torch.cuda.mem_get_info(device)[1] / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total), device)
    run = Path(args.out) / f'fold{args.fold}'
    if rt.primary:
        run.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    if (run / 'done.json').exists():
        return
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    tr = pd.read_csv(Path(args.data) / 'train.csv', dtype=str)
    classes = sorted(tr.model.unique()); n_official = len(tr)
    ex = pd.read_parquet(args.stream, columns=['text', 'model', 'prompt', 'w'])
    assert ex.model.isin(classes).all() and not ex.text.isin(set(tr.text.values)).any()
    texts = np.concatenate([tr.text.values, ex.text.values]); y = np.concatenate([np.searchsorted(classes, tr.model.values), np.searchsorted(classes, ex.model.values)])
    prompts = np.concatenate([tr.prompt_id.values.astype(str), ex.prompt.values.astype(str)])
    gid = pd.factorize(prompts)[0].astype(np.int64)
    wts = np.concatenate([np.ones(n_official, np.float32), ex.w.values.astype(np.float32)])
    lab_of = {i: LAB.get(c.split('-')[0], 3) for i, c in enumerate(classes)}
    folds = {int(k): np.array(v) for k, v in json.loads(Path(args.folds).read_text()).items()}
    val = folds[args.fold]
    rest = np.setdiff1d(np.arange(n_official), val)
    inner_p = inner_split(prompts[rest], args.inner_prompts, args.seed)
    inner = np.array([r for r in rest if prompts[r] in inner_p], int); off_rows = np.setdiff1d(rest, inner)
    pool_rows = np.arange(n_official, len(texts))
    assert not set(prompts[val]) & set(prompts[off_rows]) and not set(prompts[inner]) & set(prompts[off_rows])

    def build_pass(pass_idx):
        rng = np.random.default_rng([args.seed, pass_idx])
        pc = pool_cells(prompts, y, pool_rows, lab_of, rng, args.batch, args.focus_models, args.p_family) if len(pool_rows) else []
        stream, off = [], []
        for i, b in enumerate(pc):
            stream.append(b)
            if args.official_every > 0 and (i + 1) % (args.official_every - 1) == 0:
                if not off:
                    off = official_cells(prompts, y, off_rows, 4, 4, args.batch, rng)
                stream.append(off.pop())
        return stream

    def build_stage2():
        rng = np.random.default_rng([args.seed, 1000])
        return [c for _ in range(args.stage2_epochs) for c in official_cells(prompts, y, off_rows, 4, 4, args.batch, rng)]

    stream = build_pass(0)
    cnt = np.zeros(len(classes))
    for b in stream:
        np.add.at(cnt, y[b], wts[b])
    prior = cnt / cnt.sum(); log_prior = torch.tensor(args.logit_adjust * np.log(prior), dtype=torch.float32, device=device)
    s2 = build_stage2(); cnt2 = np.zeros(len(classes))
    for b in s2:
        np.add.at(cnt2, y[b], 1.0)
    log_prior2 = torch.tensor(args.logit_adjust * np.log(cnt2 / cnt2.sum()), dtype=torch.float32, device=device)
    log(json.dumps(dict(pool_rows=len(pool_rows), official_rows=len(off_rows), inner_rows=len(inner), inner_prompts=len(inner_p), val_rows=len(val),
                          batches_pass0=len(stream), stage2_batches=len(s2), prior_min=round(float(prior.min()), 4), prior_max=round(float(prior.max()), 4))), flush=True)
    if rt.primary:
        np.save(run / 'inner_rows.npy', inner); np.save(run / 'val_rows.npy', val)


    torch.manual_seed(args.seed)
    tok, model = model_factory(args, len(classes), device)
    params = [p for p in model.parameters() if p.requires_grad]
    if rt.world > 1:
        for parameter in params:
            torch.distributed.broadcast(parameter.data, 0)
    torch.manual_seed(args.seed + rt.rank)
    np.random.seed(args.seed + rt.rank)
    ids_cache = {}

    def ids_of(rows):
        missing = [int(r) for r in rows if int(r) not in ids_cache]
        if missing:
            for row, value in zip(missing, encode_first_window(tok, texts[missing])):
                ids_cache[row] = value
        return ids_cache

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    train_end = datetime.fromisoformat(args.train_end).timestamp()
    stage2_end = datetime.fromisoformat(args.stage2_end).timestamp()
    started = rt.scalar(time.time())
    st = dict(stage=1, elapsed=0.0, budget=max(1.0, train_end - started),
              pass_idx=0, pos=0, pos2=0, step=0, step2=0, rows=0,
              rows_official=0, evals=[], history=[], micro=args.micro_batch,
              resumes=0, stage2_stopped_by_deadline=False)
    ck = run / 'checkpoint.pt'
    if ck.exists():
        saved = torch.load(ck, map_location='cpu', weights_only=False)
        if saved.get('world_size', 1) != rt.world:
            raise ValueError('Resume requires the original checkpoint world size')
        missing, unexpected = model.load_state_dict(saved['model'], strict=False)
        assert not unexpected and not [k for k in missing if k in trainable_state(model)]
        opt.load_state_dict(saved['opt']); st = saved['st']; st['resumes'] += 1
        rt.restore_rng(saved['rng_states'])
        st['budget'] = max(st['elapsed'], min(st['budget'], st['elapsed'] + train_end - started))
        if st['pass_idx'] > 0:
            stream = build_pass(st['pass_idx'])
        log(json.dumps(dict(resumed=True, stage=st['stage'], step=st['step'])), flush=True)
    elif train_end - started < 600 and not args.max_steps:
        raise SystemExit('Fresh start requires at least ten minutes of pool-stage budget')
    t_run, el0 = started, st['elapsed']
    last_ck = started

    def save_ckpt():
        st['elapsed'] = el0 + rt.scalar(time.time()) - t_run
        rng = rt.checkpoint_rng()
        if rt.primary:
            atomic_save(dict(model=trainable_state(model), opt=opt.state_dict(),
                             st=st, rng_states=rng, world_size=rt.world), ck)
        rt.barrier()

    def evaluate(tag):
        ids_of(val)
        pv = parallel_predict(model, tok, [ids_cache[int(r)] for r in val], device, predict, rt)
        if rt.primary:
            np.save(run / f'val_probs_{tag}.npy', pv)
        rec = dict(tag=tag, step=st['step'] + st['step2'], rows=st['rows'],
                   val_balanced_accuracy=float(balanced_accuracy_score(y[val], pv.argmax(1))),
                   val_nll=float(-np.log(pv[np.arange(len(val)), y[val]] + 1e-12).mean()))
        st['evals'].append(rec)
        log(json.dumps(rec), flush=True)

    def step_on(cell, prior):
        ids_of(cell)
        while True:
            try:
                if rt.world > 1:
                    return parallel_step(model, tok, ids_cache, y, wts, gid, cell,
                                         device, args, st['micro'], prior, cell_loss, rt)
                return train_step(model, tok, ids_cache, y, wts, gid, cell, device,
                                  args, st['micro'], prior)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if st['micro'] == 1:
                    raise
                st['micro'] = max(1, st['micro'] // 2)
                log(json.dumps(dict(oom_retry=True, micro_batch=st['micro'])), flush=True)

    model.train()
    while st['stage'] == 1:
        now = rt.scalar(time.time())
        progress = (el0 + now - t_run) / st['budget']
        if progress >= 1 or (args.max_steps and st['step'] >= args.max_steps):
            st['stage'] = 2
            save_ckpt()
            if args.monitor:
                evaluate('stage1_end')
            break
        if rt.any(stop['flag']):
            save_ckpt()
            raise SystemExit(75)
        if st['pos'] >= len(stream):
            st['pass_idx'] += 1; st['pos'] = 0
            stream = build_pass(st['pass_idx'])
        cell = stream[st['pos']]
        for group in opt.param_groups:
            group['lr'] = args.lr * lr_factor(progress, args.warmup)
        parts = step_on(cell, log_prior)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        st['pos'] += 1; st['step'] += 1; st['rows'] += len(cell)
        st['rows_official'] += sum(int(r < n_official) for r in cell)
        if st['step'] % 50 == 0:
            log(json.dumps(dict(stage=1, step=st['step'], rows=st['rows'],
                                progress=round(progress, 4), ce=parts[0], world_size=rt.world)), flush=True)
        if args.monitor:
            for q in (25, 50, 75):
                if progress * 100 >= q and not any(e['tag'] == f'p{q}' for e in st['evals']):
                    evaluate(f'p{q}')
        if now - last_ck >= 60 * args.ckpt_min:
            save_ckpt(); last_ck = now
    n2 = len(s2) if not args.max_steps2 else min(len(s2), args.max_steps2)
    st['stage2_stopped_by_deadline'] = False
    while st['pos2'] < n2:
        now = rt.scalar(time.time())
        if rt.any(stop['flag']) or now > stage2_end:
            st['stage2_stopped_by_deadline'] = now > stage2_end
            save_ckpt()
            log('Saved incomplete stage 2; resume before exporting a deployable state', flush=True)
            raise SystemExit(75)
        for group in opt.param_groups:
            group['lr'] = args.lr2 * lr_factor(st['pos2'] / n2, 0.05)
        cell = s2[st['pos2']]
        parts = step_on(cell, log_prior2)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        st['pos2'] += 1; st['step2'] += 1; st['rows'] += len(cell)
        st['rows_official'] += len(cell)
        if st['step2'] % 25 == 0:
            log(json.dumps(dict(stage=2, step=st['step2'], of=n2, ce=parts[0])), flush=True)
        if now - last_ck >= 60 * args.ckpt_min:
            save_ckpt(); last_ck = now
    save_ckpt(); evaluate('final')
    if rt.primary:
        export_public(model, run / 'trainables.safetensors')
        record = dict(args=vars(args), classes=classes, name='11/11', seed=args.seed,
                      world_size=rt.world, steps_stage1=st['step'], steps_stage2=st['step2'],
                      stage2_planned=n2, stage2_stopped_by_deadline=False,
                      rows_trained=st['rows'], rows_official=st['rows_official'],
                      inner_prompts=sorted(inner_p), resumes=st['resumes'],
                      micro_batch_final=st['micro'], evals=st['evals'])
        temporary = run / 'done.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, run / 'done.json')
        ck.unlink(missing_ok=True)
        log('done', run, flush=True)
    rt.barrier()


def main(argv=None, model_factory=build_model):
    from distributed import Runtime
    args = parse(argv)
    rt = Runtime(args.device)
    try:
        return _run(args, rt, model_factory)
    finally:
        rt.close()


if __name__ == '__main__':
    main()
