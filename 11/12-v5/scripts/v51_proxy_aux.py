"""CPU optional: record historical leaderboard/proxy pairs and fit an auxiliary affine map.
Missing scores remain pending. This script never writes decision.json or changes component gates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from v51_common import ROOT, RUN, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, default=ROOT / 'artifacts/v51_leaderboard.json')
    p.add_argument('--out', type=Path, default=RUN / 'evidence/proxy_aux.json')
    a = p.parse_args(); source = json.loads(a.observations.read_text())
    if source['scale'] != '0_to_1':
        raise ValueError('Record all scores on the same 0..1 scale')
    rows = source['observations']; complete = [r for r in rows if r.get('proxy') is not None and r.get('leaderboard') is not None]
    result = dict(scope='auxiliary_only', affects_acceptance=False, observations_sha256=sha(a.observations),
                  complete=len(complete), pending=[r['run'] for r in rows if r not in complete])
    if len(complete) >= 3 and np.std([r['proxy'] for r in complete]) > 1e-8:
        x = np.array([r['proxy'] for r in complete]); y = np.array([r['leaderboard'] for r in complete])
        if not np.isfinite(x).all() or not np.isfinite(y).all() or min(x.min(), y.min()) < 0 or max(x.max(), y.max()) > 1:
            raise ValueError('Invalid recorded scores')
        design = np.column_stack([np.ones(len(x)), x]); fit = np.linalg.lstsq(design, y, rcond=None)[0]
        errors = []
        for k in range(len(x)):
            tr = np.arange(len(x)) != k
            model = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
            errors.append(float(design[k] @ model - y[k]))
        result.update(status='FITTED_AUXILIARY', intercept=float(fit[0]), slope=float(fit[1]),
                      leave_one_out_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                      warning='few correlated public submissions; this fit is not validation or hidden-test evidence')
    else:
        result['status'] = 'PENDING_SCORES'
    dump(a.out, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
