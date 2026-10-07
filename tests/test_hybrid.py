import pytest

from src.dense import HybridRetriever
from src.sparse import Hit


class Fixed:
    def __init__(self, name, ids):
        self.name, self.ids = name, ids

    def search(self, query, k=10, explain=False):
        return [Hit(d, 1.0) for d in self.ids[:k]]


def test_rrf_fuses_ranks():
    h = HybridRetriever([Fixed("a", ["x", "y", "z"]), Fixed("b", ["y", "w"])], k=60)
    hits = h.search("q", k=4, explain=True)
    assert [x.doc_id for x in hits] == ["y", "x", "w", "z"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0].terms == {"a": pytest.approx(1 / 62), "b": pytest.approx(1 / 61)}
