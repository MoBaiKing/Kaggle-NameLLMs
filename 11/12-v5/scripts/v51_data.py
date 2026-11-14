"""Refresh/audit public sources, remove official overlap; no class cap.
Unverified Gemini rows are quarantined until three independently identified model scores agree.
"""
import argparse
import json
import subprocess
import unicodedata

import numpy as np
import pandas as pd

from v51_common import BASE, ROOT, CLASSES, dump, sha, load_verified


def keys(texts):
    return [' '.join(unicodedata.normalize('NFKC', str(t)).split()) for t in texts]


def screen(frame, path):
    rec = load_verified(path)
    if rec.get('source_sha256') != sha(ROOT / 'cache/open_pool_latest/gemini37_unverified.parquet'):
        raise ValueError('Screening scores refer to a different unverified file')
    if len(set(rec['model_ids'])) < 3 or len(rec['model_ids']) != 3:
        raise ValueError('Need three independent model identities, not three folds of one model')
    with np.load(path, allow_pickle=False) as z:
        probs, ids, labels = z['probs'], z['ids'].astype(str), z['labels'].tolist()
    if labels != CLASSES or not np.array_equal(ids, frame.id.astype(str)):
        raise ValueError('Screen label/row order changed')
    if probs.shape != (3, len(frame), 18) or not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError('Bad screening probabilities')
    if not np.allclose(probs.sum(-1), 1, atol=1e-5):
        raise ValueError('Screen scores must be normalized probabilities')
    target = CLASSES.index('gemini-3.7-flash')
    return ((probs.argmax(-1) == target).sum(0) >= 2) & (probs[:, :, target].mean(0) >= .8)


def clean(frame, official):
    required = ['id', 'text', 'model', 'prompt_canon', 'source_dataset']
    if not set(required) <= set(frame):
        raise ValueError('Source must preserve row IDs, recovered prompts and provenance')
    frame = frame[frame.model.isin(CLASSES)].copy()
    frame['text_key'] = keys(frame.text)
    frame = frame[~frame.text_key.isin(set(keys(official.text)))]
    # Known topic overlap/routing flags are not training evidence.
    for flag in ('flag_near_train_prompt', 'flag_train_prompt_regen', 'flag_gemini37_routing_risk',
                 'flag_kb_proxy_after_ruling'):
        if flag in frame:
            frame = frame[~frame[flag].fillna(False if flag == 'flag_kb_proxy_after_ruling' else True).astype(bool)]
    # Reject conflicting duplicate labels instead of keeping an arbitrary label.
    conflict = frame.groupby('text_key').model.transform('nunique') > 1
    frame = frame[~conflict].drop_duplicates('text_key').copy()
    frame['prompt'] = frame.prompt_canon.fillna('').astype(str)
    unknown = frame.prompt.eq('')
    frame.loc[unknown, 'prompt'] = 'unknown:' + frame.loc[unknown, 'id'].astype(str)
    frame['label'] = np.searchsorted(CLASSES, frame.model)
    frame['id'] = ['external:' + sha_text(t) for t in frame.text_key]
    return frame.drop(columns='text_key').reset_index(drop=True)


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--kaggle', default='kaggle')
    p.add_argument('--screen', type=__import__('pathlib').Path)
    a = p.parse_args(); folder = ROOT / 'cache/open_pool_latest'; folder.mkdir(parents=True, exist_ok=True)
    if a.download:
        for file in ('ntl_open_pool.parquet', 'gemini37_unverified.parquet'):
            subprocess.run([a.kaggle, 'datasets', 'download', '-d', 'crtproduction/name-that-llm-open-pool',
                            '-f', file, '-p', str(folder), '--unzip'], check=True)
    plan = json.loads((ROOT / 'artifacts/v51_plan.json').read_text())
    for name, expected in plan['pool_files'].items():
        if sha(folder / name) != expected:
            raise ValueError('Public pool changed: audit the update and freeze a new manifest before using it')
    official = pd.read_csv(BASE / 'data/original/train.csv', dtype=str)
    source = pd.read_parquet(folder / 'ntl_open_pool.parquet')
    extra = pd.read_parquet(folder / 'gemini37_unverified.parquet')
    n_screened = 0
    if a.screen:
        accepted = screen(extra, a.screen)
        extra = extra[accepted].copy()
        # Screening overrides this quarantine flag only; other exclusion flags still apply.
        extra['flag_gemini37_routing_risk'] = False
        n_screened = len(extra); source = pd.concat([source, extra], ignore_index=True)
    data = clean(source, official)
    # Additional conservative topic exclusion follows the existing v4 CPU text similarity protocol.
    import shared  # noqa: F401
    from external_v4 import prompt_similarity
    similarity = prompt_similarity(official, data.text)
    data = data[similarity < .30].reset_index(drop=True)
    # Deterministic prompt-disjoint external calibration partition (not official validation).
    buckets = np.array([int(sha_text(g)[:8], 16) % 20 for g in data.prompt])
    data['split'] = np.where(buckets == 0, 'calibration', 'train')
    out = ROOT / 'cache/external_v51.parquet'; data.to_parquet(out, index=False)
    dump(out.with_suffix('.json'), dict(payload_sha256=sha(out), official_sha256=sha(BASE / 'data/original/train.csv'),
         source_sha256=plan['pool_files'], code_sha256=sha(__file__), cap_per_class=None,
         rows=len(data), prompts=int(data.prompt.nunique()), screened_added=n_screened,
         quarantined=int(len(pd.read_parquet(folder / 'gemini37_unverified.parquet')) - n_screened),
         per_class=data.model.value_counts().sort_index().to_dict(),
         screen_sha256=sha(a.screen) if a.screen else None))
    print(out.with_suffix('.json').read_text())


if __name__ == '__main__':
    main()
