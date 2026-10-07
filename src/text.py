"""Module 1 — text processing: tokenisation, case folding, stop words, Porter stemming.

Every step is a toggle so the ablation in src/eval.py can switch them off one at a time.
The same processor must be used for documents and queries; the index stores its settings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from nltk.stem.porter import PorterStemmer

# NLTK's English stop-word list (nltk.corpus.stopwords), inlined so no corpus download is needed.
STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren aren't as at be because been
before being below between both but by can couldn couldn't d did didn didn't do does doesn
doesn't doing don don't down during each few for from further had hadn hadn't has hasn hasn't
have haven haven't having he he'd he'll her here hers herself he's him himself his how i i'd if
i'll i'm in into is isn isn't it it'd it'll it's its itself i've just ll m ma me mightn mightn't
more most mustn mustn't my myself needn needn't no nor not now o of off on once only or other our
ours ourselves out over own re s same shan shan't she she'd she'll she's should shouldn
shouldn't should've so some such t than that that'll the their theirs them themselves then there
these they they'd they'll they're they've this those through to too under until up ve very was
wasn wasn't we we'd we'll we're were weren weren't we've what when where which while who whom why
will with won won't wouldn wouldn't y you you'd you'll your you're yours yourself yourselves you've
""".split())

# Words: letters/digits, optionally joined by an apostrophe (IRA's, don't).
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(token: str) -> str:
    return _stemmer.stem(token)


@dataclass(frozen=True)
class TextProcessor:
    lowercase: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_config(cls, cfg: dict) -> TextProcessor:
        t = cfg["text"]
        return cls(t["lowercase"], t["remove_stopwords"], t["stem"])

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text)

    def normalize(self, token: str) -> str | None:
        """One token through the pipeline; None if it is a stop word."""
        if self.lowercase:
            token = token.lower()
        if self.remove_stopwords and token.lower() in STOPWORDS:
            return None
        if token.endswith("'s"):
            token = token[:-2]
        if self.stem:
            token = _stem(token)
        return token or None

    def process(self, text: str) -> list[str]:
        """Text → index terms, in order (positions are list indices)."""
        out = []
        cache = self._cache
        for tok in self.tokenize(text):
            if tok not in cache:
                cache[tok] = self.normalize(tok)
            term = cache[tok]
            if term is not None:
                out.append(term)
        return out

    @property
    def settings(self) -> tuple[bool, bool, bool]:
        return (self.lowercase, self.remove_stopwords, self.stem)
