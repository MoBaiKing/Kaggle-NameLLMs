"""Fixed fold candidates and prompt-cluster paired bootstrap."""
import numpy as np

CANDIDATES = [('ours4', (0, 1, 2, 3), .8), ('ours2', (0, 1), .4)]


def ba(y, predictions):
    return float(np.mean([(predictions[y == c] == c).mean() for c in np.unique(y)]))


def assess(y, groups, public, ours, fraction, draws=2000):
    y, groups = np.asarray(y), np.asarray(groups)
    if public.shape != ours.shape or public.shape[0] != len(y):
        raise ValueError('Mismatched held-out predictions')
    if not np.isfinite(public).all() or not np.isfinite(ours).all():
        raise ValueError('Non-finite predictions')
    base = public.argmax(1)
    candidate = ((1 - fraction) * public + fraction * ours).argmax(1)
    delta = ba(y, candidate) - ba(y, base)
    guard = ba(y, ours.argmax(1)) - ba(y, base)
    indices = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(draws):
        rows = np.concatenate([indices[j] for j in rng.integers(0, len(indices), len(indices))])
        differences.append(ba(y[rows], candidate[rows]) - ba(y[rows], base[rows]))
    # Two fixed candidates: Bonferroni 97.5% intervals (95% family-wise target).
    lower, upper = np.quantile(differences, [.0125, .9875])
    return dict(rows=len(y), ba=100 * ba(y, candidate), public_ba=100 * ba(y, base),
                delta=100 * delta, ci97_5=[100 * float(lower), 100 * float(upper)],
                standalone_delta=100 * guard,
                fixed=int(((candidate == y) & (base != y)).sum()),
                harmed=int(((candidate != y) & (base == y)).sum()),
                passed=bool(delta > 0 and lower > 0 and guard >= 0))


def choose(reports):
    # Preference is fixed before training, rather than choosing the largest
    # observed uplift across candidates evaluated on different fold subsets.
    for name, folds, fraction in CANDIDATES:
        if reports.get(name, {}).get('passed', False):
            return name, list(folds)
    raise ValueError('No validated optimized candidate is available')
