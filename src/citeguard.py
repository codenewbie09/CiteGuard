"""Module 6 — CiteGuard: verify every answer sentence against the chunk it cites.

For each (sentence, cited chunk):
  support = alpha · tfidf_cosine(sentence, chunk) + (1 − alpha) · term_coverage
If support < threshold, the sentence is run as a BM25 query over the other retrieved chunks;
if the best one passes the threshold the citation is re-attributed, otherwise the sentence is
UNSUPPORTED. Trust = fraction of sentences that end up supported (SUPPORTED or REATTRIBUTED).

Scorers are swappable: LexicalScorer (our IR pipeline) or NLIScorer (cross-encoder entailment).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from src.index import InvertedIndex
from src.sparse import bm25_idf, bm25_term

SUPPORTED, REATTRIBUTED, UNSUPPORTED = "SUPPORTED", "REATTRIBUTED", "UNSUPPORTED"


@dataclass
class Support:
    score: float
    cosine: float = 0.0
    coverage: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)   # shared term → cosine contribution


@dataclass
class Verdict:
    sentence: str
    cited: int | None          # 1-based chunk number the LLM cited
    final: int | None          # chunk number after verification (None if unsupported)
    status: str
    score: float               # support of the final (or best-tried) chunk
    cited_score: float         # support of the originally cited chunk
    terms: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    sentences: list[Verdict]

    @property
    def trust(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(v.status != UNSUPPORTED for v in self.sentences) / len(self.sentences)


class LexicalScorer:
    name = "lexical"

    def __init__(self, index: InvertedIndex, alpha: float = 0.5):
        self.index, self.alpha = index, alpha
        self.proc = index.processor

    def _idf(self, term: str) -> float:
        # Terms the corpus never saw get the maximum idf, so invented content lowers the cosine.
        return math.log10(self.index.N / max(self.index.df(term), 1))

    def _ltc(self, terms: list[str]) -> dict[str, float]:
        w = {t: (1 + math.log10(tf)) * self._idf(t) for t, tf in Counter(terms).items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {t: x / norm for t, x in w.items()}

    def score(self, sentence: str, chunk: str) -> Support:
        s_terms = self.proc.process(sentence)
        if not s_terms:
            return Support(0.0)
        c_terms = self.proc.process(chunk)
        sv, cv = self._ltc(s_terms), self._ltc(c_terms)
        contrib = {t: sv[t] * cv[t] for t in sv if t in cv}
        cosine = sum(contrib.values())
        coverage = len(set(s_terms) & set(c_terms)) / len(set(s_terms))
        return Support(self.alpha * cosine + (1 - self.alpha) * coverage, cosine, coverage, contrib)


class NLIScorer:
    """Entailment probability P(chunk ⊨ sentence) from an NLI cross-encoder.
    This is a model judgment call by design: it is the learned baseline CiteGuard is compared to."""
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        labels = self.model.model.config.id2label
        self.entail = next(i for i, l in labels.items() if l.lower().startswith("entail"))

    def scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch: pairs of (sentence, chunk) → entailment probabilities."""
        if not pairs:
            return []
        probs = self.model.predict([(c, s) for s, c in pairs], apply_softmax=True, show_progress_bar=False)
        return [float(p[self.entail]) for p in probs]

    def score(self, sentence: str, chunk: str) -> Support:
        return Support(self.scores([(sentence, chunk)])[0])


class CiteGuard:
    def __init__(self, scorer, index: InvertedIndex, threshold: float, k1: float = 1.2, b: float = 0.75):
        self.scorer, self.index, self.threshold = scorer, index, threshold
        self.k1, self.b = k1, b

    def best_alternative(self, sentence: str, chunks: list[str], exclude: int | None) -> int | None:
        """BM25 over the retrieved chunks (minus the cited one), with corpus idf and avgdl.
        Returns the 1-based number of the best chunk, or None if no chunk shares a term."""
        idx, proc = self.index, self.index.processor
        q = list(dict.fromkeys(proc.process(sentence)))
        avgdl = idx.avgdl("body")
        best, best_score = None, 0.0
        for n, chunk in enumerate(chunks, 1):
            if n == exclude:
                continue
            terms = proc.process(chunk)
            tf = Counter(terms)
            s = sum(bm25_term(tf[t], len(terms), avgdl, bm25_idf(idx.N, idx.df(t)), self.k1, self.b)
                    for t in q if tf[t])
            if s > best_score:
                best, best_score = n, s
        return best

    def check(self, pairs: list[tuple[str, int | None]], chunks: list[str]) -> Report:
        verdicts = []
        for sentence, cited in pairs:
            valid = cited is not None and 1 <= cited <= len(chunks)
            sup = self.scorer.score(sentence, chunks[cited - 1]) if valid else Support(0.0)
            if sup.score >= self.threshold:
                verdicts.append(Verdict(sentence, cited, cited, SUPPORTED, sup.score, sup.score, sup.terms))
                continue
            alt = self.best_alternative(sentence, chunks, exclude=cited if valid else None)
            alt_sup = self.scorer.score(sentence, chunks[alt - 1]) if alt else Support(0.0)
            if alt and alt_sup.score >= self.threshold:
                verdicts.append(Verdict(sentence, cited, alt, REATTRIBUTED, alt_sup.score, sup.score, alt_sup.terms))
            else:
                verdicts.append(Verdict(sentence, cited, None, UNSUPPORTED, sup.score, sup.score, sup.terms))
        return Report(verdicts)
