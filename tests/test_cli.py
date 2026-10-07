import pytest

from src.cli import find_repairable_swap
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
    docs = [{"id": str(i), "title": "", "body": c} for i, c in enumerate(CHUNKS)]
    idx = InvertedIndex.build(docs, TextProcessor())
    return CiteGuard(LexicalScorer(idx, 0.5), idx, 0.35)


def test_finds_a_swap_that_citeguard_repairs(guard):
    answers = [{"question": "q", "chunks": CHUNKS,
                "pairs": [["Gold is shiny.", 1], ["Index funds have very low expense ratios.", 2]]}]
    a, i, wrong = find_repairable_swap(answers, guard)
    assert (a, i) == (0, 1) and wrong != 2
    v = guard.check([("Index funds have very low expense ratios.", wrong)], CHUNKS).sentences[0]
    assert v.status == "REATTRIBUTED" and v.final == 2


def test_returns_none_when_nothing_is_repairable(guard):
    assert find_repairable_swap([{"question": "q", "chunks": CHUNKS, "pairs": [["Gold is shiny.", 1]]}], guard) is None


def test_clearest_swap_maximises_the_support_jump(guard):
    from src.cli import clearest_swap, repairable_swaps

    answers = [{"question": "q", "chunks": CHUNKS,
                "pairs": [["Index funds have very low expense ratios.", 2],
                          ["Dividends are paid to shareholders out of profits.", 3]]}]
    swaps = list(repairable_swaps(answers, guard))
    assert len(swaps) >= 2
    assert all(v[3] > 0 for v in swaps)
    assert clearest_swap(answers, guard) == max(swaps, key=lambda x: x[3])[:3]
