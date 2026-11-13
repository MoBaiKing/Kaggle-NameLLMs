"""B (SAGE) units and the Kaggle runtime path (CPU): typography tables normalize, features are row-relative,
the conditional-logit gradient is exact, the gate catches siblings, and run() uses A2 where a worker returned
scores and the no-A2 weights elsewhere (including when the A2 worker fails)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sage as S
import families as F

FAKE_WORKER = '''import json, sys, numpy as np
r = json.load(open(sys.argv[1])); rows = np.asarray(r['rows'])
cand = np.load(r['candidates'])[rows]
sums = np.where(cand == cand[:, [1]], 0., -50.)          # A2 says: the 2nd candidate is right
np.savez(r['out'], rows=rows, sums=sums, tokens=np.full(len(rows), 100.), done=np.ones(len(rows), bool))
'''


def test(root):
    rng = np.random.default_rng(0)
    texts = ['Plain text — with an em dash; and “quotes”.', "Another one: it’s (fine).", 'x ' * 200, 'simple text']
    table = S.fit_typography(texts * 30, np.arange(120) % 18, 18)
    lj = np.exp(np.asarray(table['log_joint']))
    assert np.allclose(lj.sum(-1), 1)
    lp = S.typography_lp(table, texts)
    assert lp.shape == (4, 18) and np.isfinite(lp).all()
    probs = rng.dirichlet(np.ones(18) * .3, 60)
    cand = S.candidates(probs)
    assert (probs[np.arange(60), cand[:, 0]] == probs.max(1)).all()
    X, scale = S.features(probs, lp[np.zeros(60, int)], cand, F.NEIGHBORS, rng.normal(size=(60, 4)) * 5, np.full(60, 100.))
    assert X.shape == (60, 4, len(S.FEATURES_A2)) and np.allclose(X[:, 0, 0], 0) and (X[:, :, 6] <= 0).all()
    assert np.allclose(X[:, 1, 1], 1) and np.allclose(X[:, 0, 1:4], 0)
    target = rng.integers(0, 4, 60); w = rng.normal(size=X.shape[-1]); C = .5
    def f(v):
        ls = S.lsm(X @ v); return -ls[np.arange(60), target].sum() + .5 * v @ v / C
    num = np.array([(f(w + 1e-6 * e) - f(w - 1e-6 * e)) / 2e-6 for e in np.eye(len(w))])
    fitted = S.fit_logit(X, target, C)
    grad_at_opt = np.array([(f(fitted + 1e-6 * e) - f(fitted - 1e-6 * e)) / 2e-6 for e in np.eye(len(w))])
    assert np.abs(grad_at_opt).max() < 1e-3 and np.isfinite(num).all()
    sib = np.zeros((1, 18)); sib[0, F.CLASSES.index('gemini-3.7-flash')] = .9; sib[0, F.CLASSES.index('gemini-3.8-flash')] = .05; sib += 1e-4
    far = np.zeros((1, 18)); far[0, F.CLASSES.index('gemini-3.7-flash')] = .9; far[0, F.CLASSES.index('kimi-k3')] = .05; far += 1e-4
    both = np.vstack([sib, far] * 10)
    g = S.gate(both, S.candidates(both), F.NEIGHBORS)
    assert g[0] and g.reshape(10, 2)[:, 0].all()
    # runtime: a bundle that trusts A2 strongly on gated rows
    bundle = dict(classes=F.CLASSES, k=4, gate_q=1.0, neighbors=F.NEIGHBORS.astype(int).tolist(), typography=table,
                  w_no_a2=[1., 0, 0, 0, 0, 0], w_a2=[0., 0, 0, 0, 0, 0, 5., 0])
    (root / 'b.json').write_text(json.dumps(bundle))
    test_df = pd.DataFrame(dict(id=np.arange(60), text=[texts[i % 4] for i in range(60)])); test_df.to_csv(root / 'test.csv', index=False)
    (root / 'worker.py').write_text(FAKE_WORKER)
    written = {}
    out = S.run(probs, test_df, str(root / 'test.csv'), lambda *a: None, lambda p, tag: written.update(p=p, tag=tag),
                F.CLASSES, str(root / 'b.json'), 'unused', 'unused', str(root / 'worker.py'), str(root / 'work'), 1e18)
    assert (out == cand[:, 1]).all() and written['tag'] == 'dyl v4 A1+A2+B'
    (root / 'broken.py').write_text('raise SystemExit(2)\n')
    out2 = S.run(probs, test_df, str(root / 'test.csv'), lambda *a: None, lambda p, tag: None,
                 F.CLASSES, str(root / 'b.json'), 'unused', 'unused', str(root / 'broken.py'), str(root / 'work2'), 1e18)
    assert (out2 == cand[:, 0]).all(), 'failed A2 must fall back to the no-A2 re-ranker, not crash'
    print(json.dumps(dict(typography_normalized=True, row_relative_features=True, logit_gradient=True, sibling_gate=True,
                          runtime_uses_a2=True, runtime_a2_failure_uses_no_a2_weights=True)))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='v4_sage_') as d:
        test(Path(d))
