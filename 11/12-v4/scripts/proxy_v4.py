"""LB-anchored validation proxy for v4 (CPU).

The three-member LoRA OOF parent is much weaker relative to the native expert than the real public stack: on the
proxy, BA keeps rising with the native share up to a plateau at 0.70-0.825 (smoothed peak 0.725), whereas public
leaderboard ablations of the 0.684 recipe put the real optimum at the deployed 0.40 (0.30 and 0.50 both score
lower). A proxy evaluated at 0.40 therefore rewards any change that 'trusts native more' (a B re-ranker gained
+2.63 BA there purely through a native feature, and nothing once anchored). v4 evaluates, fits and trains against
the proxy at ANCHOR_SHARE = 0.75, the centre of the proxy's own optimum plateau, i.e. the proxy point that plays
the role of the deployed 0.40. Fixed before any v4 training (analysis/proxy_anchor.py, b_anchored.py).
Deployment is unchanged: the notebook still mixes native at 0.40.
"""
import shared  # noqa: F401  (import paths)
import proxy_v3 as X

ANCHOR_SHARE = .75


def deploy(parent_lp, native_lp, prior, share=ANCHOR_SHARE, **kw):
    return X.deploy(parent_lp, native_lp, prior, share=share, **kw)
