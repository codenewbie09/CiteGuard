import pytest

from src.index import InvertedIndex
from src.text import TextProcessor

# Toy corpus small enough to verify every score by hand (see test_sparse.py).
TOY_DOCS = [
    {"id": "d1", "title": "car insurance", "body": "car insurance auto insurance"},
    {"id": "d2", "title": "best car", "body": "best car"},
    {"id": "d3", "title": "auto repair", "body": "auto repair"},
]


@pytest.fixture
def raw_proc():
    return TextProcessor(lowercase=True, remove_stopwords=False, stem=False)


@pytest.fixture
def toy_index(raw_proc):
    return InvertedIndex.build(TOY_DOCS, raw_proc)
