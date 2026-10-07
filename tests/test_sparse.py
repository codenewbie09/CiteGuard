import math

import pytest

from src.sparse import BM25Retriever, TfidfRetriever


def test_lnc_ltc_matches_hand_computation(toy_index):
    """Query 'car insurance' on the toy corpus, worked by hand.

    query (ltc): tf=1 → 1; idf(car)=log10(3/2), idf(insurance)=log10(3); cosine-normalise.
    d1 (lnc): car tf1→1, insurance tf2→1+log10(2), auto tf1→1; cosine-normalise.
    d2 (lnc): best→1, car→1; norm sqrt(2).
    """
    q_car, q_ins = math.log10(3 / 2), math.log10(3)
    qn = math.hypot(q_car, q_ins)
    d1_ins = 1 + math.log10(2)
    d1n = math.sqrt(1 + d1_ins**2 + 1)
    exp_d1 = (q_car / qn) * (1 / d1n) + (q_ins / qn) * (d1_ins / d1n)
    exp_d2 = (q_car / qn) * (1 / math.sqrt(2))

    hits = TfidfRetriever(toy_index, zones=False).search("car insurance", k=10)
    assert [h.doc_id for h in hits] == ["d1", "d2"]
    assert hits[0].score == pytest.approx(exp_d1)  # ≈ 0.8154
    assert hits[1].score == pytest.approx(exp_d2)  # ≈ 0.2448
    assert exp_d1 == pytest.approx(0.8154, abs=1e-4)


def test_tfidf_breakdown_sums_to_score(toy_index):
    hit = TfidfRetriever(toy_index, zones=False).search("car insurance", k=1, explain=True)[0]
    assert sum(hit.terms.values()) == pytest.approx(hit.score)
    assert set(hit.terms) == {"car", "insurance"}


def test_zone_weighting_combines_title_and_body(toy_index):
    body = TfidfRetriever(toy_index, zones=False).search("repair", k=1)[0].score
    zoned = TfidfRetriever(toy_index, zones=True, w_title=0.3, w_body=0.7).search("repair", k=1)[0]
    # d3's title and body are both "auto repair", so both zone scores are equal
    assert zoned.score == pytest.approx(0.3 * body + 0.7 * body)


def test_champion_fast_path_returns_subset(toy_index):
    hits = TfidfRetriever(toy_index, zones=False, champion_r=1).search("car", k=10)
    assert len(hits) == 1


def test_bm25_matches_hand_computation(toy_index):
    """Query 'car': df=2, N=3, avgdl=8/3, k1=1.2, b=0.75; d2 is shorter so it wins."""
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 8 / 3

    def bm25(tf, dl):
        return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))

    hits = BM25Retriever(toy_index, k1=1.2, b=0.75).search("car", k=10, explain=True)
    assert [h.doc_id for h in hits] == ["d2", "d1"]
    assert hits[0].score == pytest.approx(bm25(1, 2))  # ≈ 0.5235
    assert hits[1].score == pytest.approx(bm25(1, 4))  # ≈ 0.3902
    assert hits[0].terms == {"car": pytest.approx(hits[0].score)}


def test_unknown_terms_return_nothing(toy_index):
    assert TfidfRetriever(toy_index).search("zzz", k=5) == []
    assert BM25Retriever(toy_index).search("zzz", k=5) == []
