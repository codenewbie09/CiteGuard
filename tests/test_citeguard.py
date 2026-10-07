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


from src.citeguard import anchors


def test_anchors_are_numbers_and_entities():
    a = anchors("You can take $5,000 from Roth IRAs after 59.5, a three-month rule from the IRS in Canada.")
    assert a["numbers"] == {"5000", "59.5", "3"}
    assert a["entities"] == {"roth", "ira", "irs", "canada"}


def test_sentence_initial_capital_is_not_an_entity():
    assert anchors("Dividends are taxed.")["entities"] == set()


def test_missing_anchor_lowers_support(guard):
    chunk = "Index funds charge fees of 0.1% and are sold by Vanguard."
    plain = LexicalScorer(guard.index, alpha=0.5, beta=0.0)
    penal = LexicalScorer(guard.index, alpha=0.5, beta=1.0)
    ok = "Index funds sold by Vanguard charge 0.1% fees."
    bad = "Index funds sold by Vanguard charge 2% fees."
    assert penal.score(ok, chunk).score == pytest.approx(plain.score(ok, chunk).score)
    s_plain, s_pen = plain.score(bad, chunk), penal.score(bad, chunk)
    assert s_pen.missing == ["2"]
    # 1 of 2 anchors (2, vanguard) missing → support scaled by (1 - 1.0 * 1/2)
    assert s_pen.score == pytest.approx(s_plain.score * (1 - 1 / 2))


def test_number_words_match_digits(guard):
    s = LexicalScorer(guard.index, alpha=0.5, beta=1.0).score(
        "The penalty is three months of interest.", "Early withdrawal costs 3 months interest.")
    assert s.missing == []


class FakeNLI:
    """Entailment = 0.9 if the sentence mentions 'fees', else 0.1; records what it was asked."""
    def __init__(self):
        self.seen = []

    def scores(self, pairs):
        self.seen.extend(pairs)
        return [0.9 if "fees" in s else 0.1 for s, _ in pairs]


def two_stage(guard, nli):
    return CiteGuard(guard.scorer, guard.index, guard.threshold, verifier=nli, verify_threshold=0.5)


def test_stage_two_marks_unconfirmed_claims_unverified(guard):
    g = two_stage(guard, FakeNLI())
    v = g.check([("Index funds have very low expense ratios.", 2),
                 ("Index funds have very low expense ratios and fees.", 2)], CHUNKS)
    assert [s.status for s in v.sentences] == ["UNVERIFIED", "SUPPORTED"]
    assert [s.nli for s in v.sentences] == [0.1, 0.9]
    assert v.trust == 0.5


def test_stage_two_checks_the_reattributed_chunk_and_skips_unsupported(guard):
    nli = FakeNLI()
    v = two_stage(guard, nli).check([("Index funds have very low expense ratios and fees.", 3),
                                     ("Gold always beats inflation over decades.", 1)], CHUNKS)
    assert [s.status for s in v.sentences] == ["REATTRIBUTED", "UNSUPPORTED"]
    assert nli.seen == [("Index funds have very low expense ratios and fees.", CHUNKS[1])]
    assert v.sentences[1].nli is None


def test_without_verifier_behaviour_is_unchanged(guard):
    v = guard.check([("Index funds have very low expense ratios.", 2)], CHUNKS)
    assert v.sentences[0].status == "SUPPORTED" and v.sentences[0].nli is None


def test_unicode_half_matches_decimal():
    assert anchors("You must be 59½ to withdraw.")["numbers"] == {"59.5"}
    from src.citeguard import missing_anchors
    assert missing_anchors("After 59½ you can withdraw.", "Withdrawals after age 59.5 are free.")[0] == []
