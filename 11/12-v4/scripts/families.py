"""Label structure used by A1, A2 and B. Families are read from the names; the cross-family confusable pairs
(claude-fable-5 <-> glm-5.3 / glm-5.3-flash, gpt-5.5 <-> kimi-k3) are the largest off-family error cells of the
public 0.684 rule on clean OOF (analysis/error_structure.py) and are fixed here before any v4 training."""
import numpy as np

CLASSES = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
           'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
           'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3',
           'qwen3.8-max']
CROSS_PAIRS = [('claude-fable-5', 'glm-5.3'), ('claude-fable-5', 'glm-5.3-flash'), ('gpt-5.5', 'kimi-k3')]


def family(name):
    return 'qwen' if name.startswith('qwen') else name.split('-')[0]


FAMILY = np.array([family(c) for c in CLASSES])
FAMILY_ID = np.unique(FAMILY, return_inverse=True)[1]


def neighbor_matrix(classes=CLASSES):
    """M[i, j] = 1 if j is a sibling (same family) or a fixed cross-family confusable of i (never i itself)."""
    index = {c: i for i, c in enumerate(classes)}
    fam = [family(c) for c in classes]
    M = np.zeros((len(classes), len(classes)), bool)
    for i in range(len(classes)):
        for j in range(len(classes)):
            M[i, j] = i != j and fam[i] == fam[j]
    for a, b in CROSS_PAIRS:
        M[index[a], index[b]] = M[index[b], index[a]] = True
    return M


NEIGHBORS = neighbor_matrix()
