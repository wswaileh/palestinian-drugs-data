"""Retrieval metrics + paired bootstrap for significance testing.

Inputs are *retrieved ingredient names* (one list per query, in rank order)
and *gold ingredient names* (one set per query). Working at ingredient
granularity matches the spec -- chunk-level retrieval is collapsed to its
ingredient before scoring.
"""

import math
import numpy as np


def recall_at_k(retrieved, gold, k):
    if not gold:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for g in gold if g in top)
    return hits / len(gold)


def mrr_at_k(retrieved, gold, k):
    for i, ing in enumerate(retrieved[:k], 1):
        if ing in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, gold, k):
    dcg = 0.0
    for i, ing in enumerate(retrieved[:k], 1):
        if ing in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def paired_bootstrap(a, b, n_resamples=1000, seed=0):
    """Two-sided paired bootstrap on per-query metric arrays.

    Returns dict with keys mean_diff (b-a), ci_low, ci_high, p_value (for null
    hypothesis "no difference").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    diffs = b - a
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[r] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    if mean_diff >= 0:
        p = (means <= 0).mean() * 2
    else:
        p = (means >= 0).mean() * 2
    return {"mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "p_value": float(min(p, 1.0))}
