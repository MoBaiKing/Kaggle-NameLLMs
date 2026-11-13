"""CCL-Gen: contrastive class-conditional likelihood expert (A2). Self-contained (torch + transformers +
safetensors) so the same file runs in training, OOF scoring and the offline Kaggle notebook.

A frozen causal LM (Qwen3-1.7B-Base) gets one shared LoRA (rank 16 on every attention/MLP projection) and a
learned soft prefix of PREFIX vectors per class. The class score of a text x is
    S(x, y) = sum_t log p(x_t | prefix_y, x_<t)      (text tokens only, first MAX_TOKENS tokens)
i.e. a generative classifier (ATLAS, arXiv 2610.19583; generative classifiers are less shortcut-prone).
Training mixes the generative NLL of the true class with a discriminative CE over S on a candidate set made of
the true class, its siblings / fixed confusables and random others, so likelihood mass is pushed apart exactly
where the public stack confuses versions. Per text, a constant base-LM likelihood would cancel in any comparison
between classes, so no separate base-likelihood term is computed.
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

RANK, ALPHA, PREFIX, MAX_TOKENS = 16, 32, 8, 320
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
PROBE = 'Identify the writing style: a short probe sentence, with punctuation; and digits 12345.'


class LoRALinear(nn.Module):
    def __init__(self, base, rank=RANK, alpha=ALPHA):
        super().__init__()
        self.base_layer = base
        dev = base.weight.device
        self.a = nn.Linear(base.in_features, rank, bias=False, device=dev, dtype=torch.float32)
        self.b = nn.Linear(rank, base.out_features, bias=False, device=dev, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        out = self.base_layer(x)
        return (out.float() + self.b(self.a(x.float())) * self.scale).to(out.dtype)


class CCLGen(nn.Module):
    def __init__(self, lm, n_classes, prefix=PREFIX, init_ids=None):
        super().__init__()
        self.lm = lm
        lm.requires_grad_(False)
        wrapped = 0
        for name, module in list(lm.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in TARGETS:
                parent, attr = name.rsplit('.', 1)
                setattr(lm.get_submodule(parent), attr, LoRALinear(module)); wrapped += 1
        if wrapped != lm.config.num_hidden_layers * len(TARGETS):
            raise ValueError(f'Unexpected LoRA target count {wrapped}')
        emb = lm.get_input_embeddings().weight
        if init_ids is None:
            init = emb.float().mean(0, keepdim=True).repeat(n_classes * prefix, 1)
        else:
            init = emb[torch.as_tensor(init_ids, device=emb.device).reshape(-1)].float()
        self.prefix = nn.Parameter(init.reshape(n_classes, prefix, -1).clone().to(emb.device))
        self.n_classes, self.n_prefix = n_classes, prefix

    def trainable_state(self):
        out = {'prefix': self.prefix.detach().float().cpu()}
        for name, module in self.lm.named_modules():
            if isinstance(module, LoRALinear):
                out[f'{name}.a'] = module.a.weight.detach().float().cpu()
                out[f'{name}.b'] = module.b.weight.detach().float().cpu()
        return out

    def load_trainable(self, state):
        expected = {k for k in self.trainable_state()}
        if set(state) != expected:
            raise ValueError('CCL-Gen state layout differs')
        with torch.no_grad():
            self.prefix.copy_(state['prefix'].to(self.prefix))
            for name, module in self.lm.named_modules():
                if isinstance(module, LoRALinear):
                    module.a.weight.copy_(state[f'{name}.a']); module.b.weight.copy_(state[f'{name}.b'])

    def sequence_scores(self, input_ids, attention_mask, classes, chunk=4):
        """input_ids/attention_mask [B, T] (right padded text tokens), classes [B] -> (sum log-prob [B], tokens [B])."""
        emb = self.lm.get_input_embeddings()(input_ids)
        pre = self.prefix[classes].to(emb.dtype)
        P = pre.shape[1]
        inputs = torch.cat([pre, emb], 1)
        mask = torch.cat([attention_mask.new_ones(len(classes), P), attention_mask], 1)
        hidden = self.lm.model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).last_hidden_state
        head = self.lm.get_output_embeddings()
        T = input_ids.shape[1]
        sums = []
        for s in range(0, len(classes), chunk):
            h = hidden[s:s + chunk, P - 1:P + T - 1]
            logits = head(h).float()
            lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids[s:s + chunk].reshape(-1),
                                  reduction='none').reshape(h.shape[0], T)
            sums.append((lp * attention_mask[s:s + chunk]).sum(1))
        return torch.cat(sums), attention_mask.sum(1)


def build(base_path, n_classes, device, dtype, class_names=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path)
    lm = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, attn_implementation='sdpa').to(device)
    lm.config.use_cache = False
    init_ids = None
    if class_names is not None:
        init_ids = []
        for name in class_names:
            ids = tok(f' {name} model wrote:', add_special_tokens=False)['input_ids']
            init_ids.append((ids * PREFIX)[:PREFIX])
    return tok, CCLGen(lm, n_classes, init_ids=init_ids).to(device)


def encode(tok, texts, max_tokens=MAX_TOKENS):
    return [tok(str(t), add_special_tokens=False, truncation=True, max_length=max_tokens)['input_ids'] or [tok.eos_token_id]
            for t in texts]


def pad(batch, pad_id, device):
    width = max(len(x) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def score(model, tok, encoded, candidates, device, batch_texts=4, deadline=None, log=print):
    """encoded: list of token lists; candidates: [N, K] class indices -> sums [N, K], tokens [N], done [N]."""
    import time
    import numpy as np
    N, K = len(encoded), len(candidates[0])
    sums = np.zeros((N, K)); tokens = np.zeros(N); done = np.zeros(N, bool)
    order = sorted(range(N), key=lambda i: len(encoded[i]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, N, batch_texts):
        if deadline is not None and time.time() > deadline:
            log(f'CCL-Gen deadline: {int(done.sum())}/{N} rows scored')
            break
        rows = order[start:start + batch_texts]
        seqs = [encoded[r] for r in rows for _ in range(K)]
        classes = torch.tensor([int(c) for r in rows for c in candidates[r]], device=device)
        ids, mask = pad(seqs, pad_id, device)
        s, n = model.sequence_scores(ids, mask, classes)
        s = s.float().cpu().numpy().reshape(len(rows), K)
        ok = np.isfinite(s).all(1)          # an fp16 overflow leaves that row unscored (B then uses its no-A2 variant)
        sums[rows] = np.where(ok[:, None], s, 0.); tokens[rows] = n.reshape(len(rows), K)[:, 0].cpu().numpy()
        done[np.asarray(rows)[ok]] = True
    return sums, tokens, done


def load_bundle(bundle_dir, base_path, device, dtype):
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(bundle_dir, 'a2_bundle.json')))
    tok, model = build(base_path, len(meta['classes']), device, dtype)
    if encode(tok, [PROBE])[0] != meta['probe_ids']:
        raise RuntimeError('CCL-Gen tokenizer differs from training')
    model.load_trainable(load_file(os.path.join(bundle_dir, 'adapters.safetensors')))
    model.eval()
    return meta, tok, model


def worker_main(argv):
    """Kaggle: python dyl_ccl.py <request.json>  (one GPU per process; fp16 on T4)."""
    import numpy as np
    request = json.load(open(argv[0]))
    device = torch.device('cuda:0')
    meta, tok, model = load_bundle(request['bundle'], request['base'], device, torch.float16)
    import pandas as pd
    texts = pd.read_csv(request['query']).text.fillna('').astype(str).values[request['rows']].tolist()
    candidates = np.load(request['candidates'])[request['rows']]
    sums, tokens, done = score(model, tok, encode(tok, texts, meta['max_tokens']), candidates, device,
                               request.get('batch', 4), request.get('deadline'))
    np.savez(request['out'], rows=np.asarray(request['rows']), sums=sums, tokens=tokens, done=done)


if __name__ == '__main__':
    worker_main(sys.argv[1:])
