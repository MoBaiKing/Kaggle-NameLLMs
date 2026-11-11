"""Fetch only external assets; verify pinned inputs and prepare the native tokenizer."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'config/assets.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f'Missing or changed pinned input: {path}')


def originals():
    for name, entry in SPEC['original_data'].items():
        check(ROOT / 'data/original' / name, entry['sha256'])
    check(ROOT / 'data/folds.json', SPEC['folds_sha256'])
    for name, entry in SPEC.get('reference_data', {}).items():
        check(ROOT / 'data/reference' / name, entry['sha256'])
    print('Original CSV/Parquet bytes and frozen folds verified')


def augmentation():
    entry = SPEC['augmentation']
    target = ROOT / 'data' / entry['filename']
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tag = urllib.parse.quote(entry['release_tag'], safe='')
        url = f'https://github.com/MoBaiKing/Kaggle-NameLLMs/releases/download/{tag}/{target.name}'
        temporary = target.with_suffix('.download')
        try:
            urllib.request.urlretrieve(url, temporary)
            check(temporary, entry['sha256'])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    check(target, entry['sha256'])
    print('Pinned 153,770-row augmentation stream verified')


def dataset_file(relative, destination, expected):
    if destination.exists():
        check(destination, expected)
        return
    dataset = SPEC['native_assets']['dataset']
    subdirectory = SPEC['native_assets']['subdirectory']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native_download_') as temp:
        subprocess.run(['kaggle', 'datasets', 'download', dataset,
                        '-f', f'{subdirectory}/{relative}', '-p', temp], check=True)
        # Kaggle can wrap an individual nested file in a ZIP. Extract regular
        # members under this temporary directory only.
        for z in list(Path(temp).rglob('*.zip')):
            with zipfile.ZipFile(z) as archive:
                for member in archive.infolist():
                    path = (Path(temp) / member.filename).resolve()
                    if not path.is_relative_to(Path(temp).resolve()):
                        raise ValueError('Unsafe archive member')
                archive.extractall(temp)
        candidates = [p for p in Path(temp).rglob(Path(relative).name)
                      if p.is_file() and sha(p) == expected]
        if not candidates:
            raise ValueError(f'Upstream asset changed: {relative}; locate the matching historical version using REPRODUCE_AGENT.md')
        shutil.copyfile(candidates[0], destination)
    check(destination, expected)


def native_assets():
    entry = SPEC['native_assets']; sub = entry['subdirectory']
    weights = ROOT / 'external/native_q8' / sub
    metadata = ROOT / 'external/native_q8_meta' / sub
    dataset_file('manifest.json', weights / 'manifest.json', entry['manifest_sha256'])
    manifest = json.loads((weights / 'manifest.json').read_text())
    hashes = manifest['file_hashes']
    small = ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'merges.txt', 'vocab.json']
    for name in small:
        relative = f'base/{name}'
        dataset_file(relative, weights / relative, hashes[relative])
    for fold in range(5):
        for relative in [f'calibration/fold{fold}.json', f'fold{fold}/trainables.safetensors']:
            dataset_file(relative, weights / relative, hashes[relative])
        relative = f'fold{fold}/split.json'
        dataset_file(relative, metadata / relative, hashes[relative])
    dataset_file('oof_reference.npz', metadata / 'oof_reference.npz', hashes['oof_reference.npz'])
    print('Public native fold states, calibration, splits, OOF and tokenizer verified')


def backbone():
    from huggingface_hub import snapshot_download
    entry = SPEC['model']
    path = ROOT / 'external/public_bases/qwen3-embedding-8b'
    snapshot_download(entry['repo'], revision=entry['revision'], local_dir=path,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    print('Pinned backbone downloaded:', path)


def notebook():
    destination = ROOT / 'external/pinned_notebook'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'ntl-684-ozertuu-mirror.ipynb'
    if not path.exists():
        subprocess.run(['kaggle', 'kernels', 'pull', SPEC['public_notebook']['kernel'],
                        '-p', str(destination), '-m'], check=True)
    check(path, SPEC['public_notebook']['sha256'])
    source = json.loads(path.read_text())
    vendor = ROOT / 'vendor'; vendor.mkdir(exist_ok=True)
    for cell in source['cells']:
        text = ''.join(cell.get('source', []))
        first, separator, code = text.partition('\n')
        if separator and first.startswith('%%writefile '):
            name = first.split()[-1]
            if Path(name).name != name:
                raise ValueError('Unexpected notebook output path')
            (vendor / name).write_text(code)
    check(vendor / 'native_q8.py', SPEC['public_notebook']['native_code_sha256'])
    print('Pinned upstream Notebook and native runtime verified')


def stage():
    sub = SPEC['native_assets']['subdirectory']
    native = ROOT / 'external/native_q8' / sub / 'base'
    base = ROOT / 'external/public_bases/qwen3-embedding-8b'
    target = ROOT / 'base_staged'; target.mkdir(exist_ok=True)
    index = json.loads((native / 'model.safetensors.index.json').read_text())
    for name in ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'merges.txt', 'vocab.json']:
        source = native / name
        if not source.is_file():
            raise FileNotFoundError(source)
        link = target / name
        if not link.exists():
            link.symlink_to(source.resolve())
    for name in sorted(set(index['weight_map'].values())):
        source = base / name
        if not source.is_file():
            raise FileNotFoundError(source)
        link = target / name
        if not link.exists():
            link.symlink_to(source.resolve())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target, local_files_only=True)
    assert tok('', add_special_tokens=True)['input_ids'] == [151643]
    print('Training base staged with the public end-of-text tokenizer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['all', 'verify', 'augmentation', 'backbone', 'native', 'notebook', 'stage'])
    args = parser.parse_args()
    originals()
    actions = dict(augmentation=augmentation, backbone=backbone, native=native_assets,
                   notebook=notebook, stage=stage)
    if args.action == 'all':
        for name in ['augmentation', 'backbone', 'native', 'notebook', 'stage']:
            actions[name]()
    elif args.action != 'verify':
        actions[args.action]()


if __name__ == '__main__':
    main()
