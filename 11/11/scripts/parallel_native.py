"""Shard native evaluation over four GPUs while keeping each public forward unchanged."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))


class ParallelNative:
    def __init__(self, base, assets, work, log=print, gpus=4):
        from transformers import AutoTokenizer
        self.base, self.assets, self.work = Path(base), Path(assets), Path(work)
        self.log, self.gpus = log, gpus
        self.labels = json.loads((self.assets / 'manifest.json').read_text())['classes']
        self.tokenizer = AutoTokenizer.from_pretrained(self.assets / 'base', local_files_only=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, texts, folds, batch_size=4):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        devices = visible.split(',') if visible else [str(i) for i in range(self.gpus)]
        if len(devices) < self.gpus:
            raise ValueError('Four visible GPUs are required for parallel native evaluation')
        jobs = []
        self.work.mkdir(parents=True, exist_ok=True)
        for rank in range(min(self.gpus, len(texts))):
            rows = list(range(rank, len(texts), self.gpus))
            directory = self.work / f'shard{rank}'; directory.mkdir(exist_ok=True)
            request = directory / 'request.json'
            request.write_text(json.dumps(dict(base=str(self.base), assets=str(self.assets),
                                              work=str(directory / 'native_work'),
                                              texts=[texts[i] for i in rows], folds=list(folds),
                                              batch=batch_size, rows=rows)))
            log = (directory / 'worker.log').open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank])
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        '--request', str(request)], env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((process, log, directory / 'result.npz', rows))
        try:
            for process, log, path, rows in jobs:
                if process.wait() != 0:
                    raise RuntimeError(f'Native shard failed; inspect {path.parent / "worker.log"}')
            output = dict(plain=np.zeros((len(folds), len(texts), 18)),
                          centered=np.zeros((len(folds), len(texts), 18)),
                          folds=np.asarray(list(folds)))
            finite = np.ones(len(texts), dtype=bool)
            for process, log, path, rows in jobs:
                with np.load(path, allow_pickle=False) as z:
                    assert z['rows'].tolist() == rows and z['folds'].tolist() == list(folds)
                    for name in ('plain', 'centered'):
                        output[name][:, rows, :] = z[name]
                    finite[rows] = z['finite']
                    if 'centered_bias' in output:
                        assert np.array_equal(output['centered_bias'], z['centered_bias'])
                    else:
                        output['centered_bias'] = z['centered_bias'].copy()
            return output, finite
        finally:
            for process, log, path, rows in jobs:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                log.close()


def worker(request):
    import torch
    from native_q8 import NativeQ8
    torch.cuda.set_per_process_memory_fraction(min(1.0, 28.5 * 2 ** 30 / torch.cuda.mem_get_info(0)[1]), 0)
    cfg = json.loads(request.read_text())
    model = NativeQ8(cfg['base'], cfg['assets'], cfg['work'], gpus=1)
    output, finite = model.predict(cfg['texts'], folds=cfg['folds'], batch_size=cfg['batch'])
    np.savez(request.with_name('result.npz'), rows=np.asarray(cfg['rows']), finite=finite, **output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', type=Path, required=True)
    worker(parser.parse_args().request)
