"""Ranked-retrieval metrics. `ranked` is a list of doc ids; `qrels` maps doc id → relevance grade."""
from __future__ import annotations

import math


def precision_at_k(ranked, qrels, k):
    return sum(1 for d in ranked[:k] if qrels.get(d, 0) > 0) / k


def recall_at_k(ranked, qrels, k):
    rel = sum(1 for g in qrels.values() if g > 0)
    return sum(1 for d in ranked[:k] if qrels.get(d, 0) > 0) / rel if rel else 0.0


def mrr_at_k(ranked, qrels, k):
    for i, d in enumerate(ranked[:k], 1):
        if qrels.get(d, 0) > 0:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, qrels, k):
    dcg = sum(qrels.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def paired_randomization(a, b, trials: int = 10000, seed: int = 42) -> float:
    """Two-sided paired randomisation (sign-flip) test on per-query scores; returns the p-value."""
    import numpy as np

    diff = np.asarray(a, float) - np.asarray(b, float)
    observed = abs(diff.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(trials, len(diff)))
    return float(np.mean(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12))
