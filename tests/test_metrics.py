import math

import pytest

from src.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


RANKED = ["a", "x", "b", "y", "z"]
QRELS = {"a": 1, "b": 1, "c": 1}


def test_precision_recall():
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr_at_k(RANKED, QRELS, 10) == 1.0
    assert mrr_at_k(["x", "b"], QRELS, 10) == 0.5
    assert mrr_at_k(["x", "b"], QRELS, 1) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, QRELS, 10) == pytest.approx(dcg / idcg)
