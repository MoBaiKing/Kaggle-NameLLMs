"""CPU: v3 OOF targets recomputed on the LB-anchored proxy (proxy_v4.ANCHOR_SHARE), cache/oof_targets_v4.npz.
Same parent, weights formula and checks as v3's oof_targets.py; only the native share of the deploy rule changes."""
import argparse
import json
from pathlib import Path

import shared  # noqa: F401  (import paths)
import proxy_v3 as X
import proxy_v4 as X4
import oof_targets as T
from shared import ROOT

DEFAULT = ROOT / 'cache/oof_targets_v4.npz'


def build():
    original = X.deploy
    X.deploy = lambda parent, native, prior, share=X4.ANCHOR_SHARE, **kw: original(parent, native, prior, share=share, **kw)
    try:
        payload, record = T.build()
    finally:
        X.deploy = original
    record.update(method='oof_targets_v4_anchored', native_share=X4.ANCHOR_SHARE,
                  anchor_code_sha256=T.sha(Path(X4.__file__)), builder_sha256=T.sha(Path(__file__)))
    record['inputs']['12-v4/scripts/proxy_v4.py'] = T.sha(Path(X4.__file__))
    return payload, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        meta = json.loads(a.out.with_suffix('.json').read_text())
        inputs = T.inputs(); inputs['12-v4/scripts/proxy_v4.py'] = T.sha(Path(X4.__file__))
        if meta['inputs'] != inputs or T.sha(a.out) != meta['payload_sha256']:
            raise SystemExit('v4 OOF targets are stale; rebuild them in a new cache path')
        print(json.dumps(dict(oof_targets_v4='verified', **meta['summary']))); return
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use --check or a new path')
    payload, record = build()
    record = T.write(a.out, payload, record)
    print(json.dumps(dict(oof_targets_v4=str(a.out), payload_sha256=record['payload_sha256'], **record['summary']), indent=1))


if __name__ == '__main__':
    main()
