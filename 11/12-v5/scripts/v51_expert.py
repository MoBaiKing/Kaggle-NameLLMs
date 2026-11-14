"""8B discriminative experts. Optional torch imports keep CPU protocol tests lightweight.
One group and one fold per row at deployment; no five-model voting on each test row.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from v51_common import CLASSES, GROUPS, lsm

INSTRUCTION = 'Instruct: Identify the language model that generated the text based on its writing style.\nQuery: '
SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def attach_lora(encoder):
    """Native-compatible fp32 low-rank branches; no extra Kaggle wheel dependency."""
    import math
    import torch
    from torch import nn

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__(); self.base_layer = base
            self.a = nn.Linear(base.in_features, 16, bias=False, device=base.weight.device, dtype=torch.float32)
            self.b = nn.Linear(16, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
            nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
            self.drop = nn.Dropout(.05)

        def forward(self, x):
            original = self.base_layer(x)
            return (original + 2 * self.b(self.a(self.drop(x).float()))).to(original.dtype)

    encoder.requires_grad_(False); count = 0
    for name, module in list(encoder.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in SUFFIXES:
            parent, attr = name.rsplit('.', 1)
            setattr(encoder.get_submodule(parent), attr, LoRA(module)); count += 1
    return count


def route(probs, enabled):
    """Fixed label-free route: both top candidates must belong to the same disjoint expert group."""
    cand = np.argsort(-probs, axis=1, kind='stable')[:, :2]
    result = np.full(len(probs), -1, int)
    for j, group in enumerate(GROUPS):
        if group not in enabled:
            continue
        ids = [CLASSES.index(c) for c in GROUPS[group]]
        use = np.isin(cand[:, 0], ids) & np.isin(cand[:, 1], ids)
        if (result[use] >= 0).any():
            raise ValueError('Expert groups overlap')
        result[use] = j
    return result


def deploy_fold(row_id):
    return int(hashlib.sha256(str(row_id).encode()).hexdigest()[:16], 16) % 5


def balanced_order(labels, rng):
    """Cover EVERY row; repeat small classes rather than downsampling the large ones."""
    pools = [np.flatnonzero(labels == y) for y in np.unique(labels)]
    if not pools or any(not len(x) for x in pools):
        raise ValueError('Empty expert class')
    n = max(map(len, pools))
    cols = [np.concatenate([rng.permutation(rows) for _ in range((n + len(rows) - 1) // len(rows))])[:n]
            for rows in pools]
    return np.stack(cols, 1).ravel()


def inference_device_map(layers=36):
    return {'embed_tokens': 0, 'norm': 1, 'rotary_emb': 1,
            **{f'layers.{i}': 0 if i < 16 else 1 for i in range(layers)}}


def make_model(base, n_classes, shard=False):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    class Discriminator(nn.Module):
        def __init__(self, encoder, width):
            super().__init__(); self.encoder = encoder
            self.head = nn.Linear(width, n_classes, dtype=torch.float32, device='cuda')

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            vec = out[torch.arange(len(input_ids), device=out.device), (attention_mask.sum(1) - 1).to(out.device)].float()
            return self.head(torch.nn.functional.normalize(vec, dim=-1).to(self.head.weight.device))

    tok = AutoTokenizer.from_pretrained(base, local_files_only=True); tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if not shard and torch.cuda.is_bf16_supported() else torch.float16
    if shard and torch.cuda.device_count() < 2:
        raise ValueError('Kaggle FP16 8B inference requires two GPUs; do not silently quantize')
    options = dict(device_map=inference_device_map()) if shard else {}
    encoder = AutoModel.from_pretrained(base, local_files_only=True, torch_dtype=dtype,
                                       attn_implementation='sdpa', **options)
    if not shard:
        encoder = encoder.to('cuda')
    if encoder.config.hidden_size != 4096 or encoder.config.num_hidden_layers != 36:
        raise ValueError('Expected pinned Qwen3-Embedding-8B')
    encoder.config.use_cache = False
    if attach_lora(encoder) != 36 * len(SUFFIXES):
        raise ValueError('Expert LoRA target layout differs from pinned backbone')
    if not shard:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return tok, Discriminator(encoder, encoder.config.hidden_size)


def encode(tok, texts):
    ids = tok([INSTRUCTION + str(t) for t in texts], truncation=True, max_length=511,
              add_special_tokens=False)['input_ids']
    return tok.pad({'input_ids': [x + [151643] for x in ids]}, padding=True, return_tensors='pt').to('cuda')


def state(model):
    return {n: p.detach().float().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def load_state(model, path):
    import torch
    from safetensors.torch import load_file
    values = load_file(str(path)); expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(values) != set(expected):
        raise ValueError('Expert adapter/head layout changed')
    with torch.no_grad():
        for n, p in expected.items():
            if values[n].shape != p.shape or not torch.isfinite(values[n]).all():
                raise ValueError(f'Invalid expert tensor: {n}')
            p.copy_(values[n].to(p.device, p.dtype))


def score(model, tok, texts, batch=4):
    import torch
    model.eval(); outputs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            outputs.append(model(**encode(tok, texts[i:i + batch])).float().cpu().numpy())
    result = np.concatenate(outputs) if outputs else np.empty((0, model.head.out_features))
    if not np.isfinite(result).all():
        raise ValueError('Non-finite expert scores')
    return lsm(result) if len(result) else result


def kaggle_worker(request):
    """Separate process frees GPU memory after native; only the routed rows are scored."""
    from v51_common import save_npz, sha
    req = json.loads(Path(request).read_text())
    import pandas as pd
    frame = pd.read_csv(req['query'], dtype={'id': str})
    outputs = np.full((len(frame), len(CLASSES)), np.nan)
    rows = np.array(req['rows'], int)
    for group in req['enabled']:
        use = rows[np.asarray(req['route'])[rows] == list(GROUPS).index(group)]
        if not len(use):
            continue
        tok, model = make_model(req['base'], len(GROUPS[group]), shard=True)
        ids = [CLASSES.index(c) for c in GROUPS[group]]
        for fold in range(5):
            idx = use[np.array([deploy_fold(frame.id.iloc[i]) == fold for i in use])]
            if not len(idx):
                continue
            path = Path(req['bundle']) / f'{group}_fold{fold}.safetensors'
            if sha(path) != req['hashes'][path.name]:
                raise ValueError('Expert differs from validated checkpoint')
            load_state(model, path)
            outputs[np.ix_(idx, ids)] = score(model, tok, frame.text.fillna('').iloc[idx].tolist())
        del model, tok
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    save_npz(req['out'], lp=outputs, rows=rows)


if __name__ == '__main__':
    import sys
    kaggle_worker(sys.argv[1])
