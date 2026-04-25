import math
import numpy as np
from src.eval import metrics


def test_recall_at_k_on_known_ranking():
    retrieved = ["ibuprofen", "metformin", "naproxen"]
    assert metrics.recall_at_k(retrieved, ["metformin"], k=1) == 0.0
    assert metrics.recall_at_k(retrieved, ["metformin"], k=2) == 1.0
    assert metrics.recall_at_k(retrieved, ["metformin"], k=3) == 1.0


def test_recall_with_multiple_gold():
    retrieved = ["a", "b", "c", "d"]
    assert math.isclose(
        metrics.recall_at_k(retrieved, ["b", "c", "x"], k=3), 2/3)


def test_mrr_uses_first_relevant():
    retrieved = ["a", "b", "c"]
    assert metrics.mrr_at_k(retrieved, ["c"], k=10) == 1/3
    assert metrics.mrr_at_k(retrieved, ["x"], k=10) == 0.0


def test_ndcg_perfect_ranking():
    retrieved = ["a", "b", "c"]
    assert math.isclose(metrics.ndcg_at_k(retrieved, ["a"], k=3), 1.0)


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.4, size=200)
    b = rng.uniform(0.5, 0.9, size=200)
    res = metrics.paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert res["mean_diff"] > 0
    assert res["ci_low"] > 0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_no_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.4, 0.6, size=200)
    b = rng.uniform(0.4, 0.6, size=200)
    res = metrics.paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert res["p_value"] > 0.05
