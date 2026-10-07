from src.text import TextProcessor


def test_tokenize_splits_on_punctuation_and_keeps_numbers():
    p = TextProcessor(lowercase=False, remove_stopwords=False, stem=False)
    assert p.tokenize("Roth IRA's 401k, (fees)!") == ["Roth", "IRA's", "401k", "fees"]


def test_each_step_toggles():
    text = "The Investors are Investing"
    assert TextProcessor(False, False, False).process(text) == ["The", "Investors", "are", "Investing"]
    assert TextProcessor(True, False, False).process(text) == ["the", "investors", "are", "investing"]
    assert TextProcessor(True, True, False).process(text) == ["investors", "investing"]
    assert TextProcessor(True, True, True).process(text) == ["investor", "invest"]


def test_possessive_is_stripped():
    assert TextProcessor(True, True, True).process("company's") == ["compani"]
