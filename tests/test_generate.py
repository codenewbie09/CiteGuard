from src.generate import build_prompt, parse_answer


def test_parse_one_citation_per_sentence():
    text = "Roth IRAs are funded post-tax [2]. Withdrawals are tax-free [1]."
    assert parse_answer(text) == [
        ("Roth IRAs are funded post-tax.", 2),
        ("Withdrawals are tax-free.", 1),
    ]


def test_parse_citation_after_period_and_extra_citations():
    text = "Fees matter. [3] Index funds are cheap. [1][2]\nSo pick one."
    assert parse_answer(text) == [
        ("Fees matter.", 3),
        ("Index funds are cheap.", 1),   # extra citations dropped: one per sentence
        ("So pick one.", None),          # uncited trailing sentence is kept, uncited
    ]


def test_insufficient_evidence_yields_no_claims():
    assert parse_answer("Insufficient evidence.") == []


def test_prompt_numbers_chunks_from_one():
    p = build_prompt("q?", ["alpha", "beta"])
    assert "[1] alpha" in p and "[2] beta" in p and "q?" in p
