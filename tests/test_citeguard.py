import pytest

from src.citeguard import CiteGuard, LexicalScorer
from src.index import InvertedIndex
from src.text import TextProcessor

CHUNKS = [
    "A Roth IRA is funded with after-tax money and qualified withdrawals are tax free.",
    "Index funds track a market index and usually have very low expense ratios.",
    "Dividends are paid by companies to shareholders out of profits.",
]


@pytest.fixture
def guard():
    proc = TextProcessor()
    docs = [{"id": str(i), "title": "", "body": c} for i, c in enumerate(CHUNKS)]
    docs.append({"id": "x", "title": "", "body": "unrelated text about weather and rain"})
    idx = InvertedIndex.build(docs, proc)
    return CiteGuard(LexicalScorer(idx, alpha=0.5), idx, threshold=0.35)


def test_coverage_is_fraction_of_sentence_terms_in_chunk(guard):
    s = guard.scorer.score("index funds have low fees", CHUNKS[1])
    # content terms: index, fund, low, fee → 3 of 4 present
    assert s.coverage == pytest.approx(3 / 4)
    assert set(s.terms) == {"index", "fund", "low"}


def test_correct_citation_is_supported(guard):
    v = guard.check([("Index funds have very low expense ratios.", 2)], CHUNKS)
    assert v.sentences[0].status == "SUPPORTED"
    assert v.trust == 1.0


def test_wrong_citation_is_reattributed(guard):
    v = guard.check([("Index funds have very low expense ratios.", 3)], CHUNKS)
    s = v.sentences[0]
    assert s.status == "REATTRIBUTED"
    assert (s.cited, s.final) == (3, 2)


def test_unsupported_claim_is_flagged(guard):
    v = guard.check([("Gold always beats inflation over decades.", 1),
                     ("Dividends are paid to shareholders.", 3)], CHUNKS)
    assert [s.status for s in v.sentences] == ["UNSUPPORTED", "SUPPORTED"]
    assert v.trust == 0.5


def test_missing_or_bad_citation_goes_to_reattribution(guard):
    v = guard.check([("Dividends are paid to shareholders out of profits.", None),
                     ("Dividends are paid to shareholders out of profits.", 9)], CHUNKS)
    assert [s.final for s in v.sentences] == [3, 3]
