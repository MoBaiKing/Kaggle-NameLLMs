"""Standalone NumPy voting code embedded into the offline Kaggle infer.py."""


def multi_model_parent(public_probability, terms, logger=None):
    import numpy as np
    weights = {'q14v5': 3., 'q8v5': 2., 'll8v5': 1.}
    vote = np.zeros_like(public_probability, dtype=np.float64)
    for name, weight in weights.items():
        if name not in terms:
            raise RuntimeError(f'Required strong-model voter is missing: {name}')
        logits = np.asarray(terms[name], np.float64) / weight
        if logits.shape != vote.shape or not np.isfinite(logits).all():
            raise ValueError(f'Invalid strong-model vote: {name}')
        logits -= logits.max(-1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(-1, keepdims=True)
        vote += weight * probability / sum(weights.values())
    public_probability = np.asarray(public_probability, np.float64)
    if (not np.isfinite(public_probability).all() or (public_probability < 0).any()
            or not np.allclose(public_probability.sum(-1), 1.)):
        raise ValueError('Invalid complete public ensemble probabilities')
    parent = .75 * public_probability + .25 * vote
    parent /= parent.sum(-1, keepdims=True)
    if logger is not None:
        logger('DYL_MULTI_MODEL_VOTE', weights, 'parent weights: public=.75 explicit_voters=.25')
    return parent
