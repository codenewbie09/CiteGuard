"""Module 6 — CiteGuard: verify every answer sentence against the chunk it cites.

For each (sentence, cited chunk):
  support = [alpha · tfidf_cosine(sentence, chunk) + (1 − alpha) · term_coverage] · (1 − beta · missing_anchors)
where missing_anchors is the fraction of the sentence's numbers and named entities absent from the chunk.
If support < threshold, the sentence is run as a BM25 query over the other retrieved chunks;
if the best one passes the threshold the citation is re-attributed, otherwise the sentence is
UNSUPPORTED.

Optional stage 2 (two-stage verification): every sentence that survives stage 1 is checked by an
NLI verifier against its final chunk; if entailment < verify_threshold it becomes UNVERIFIED
(the citation points at the right chunk, but the claim is not confirmed by it). Stage 1 is good
at wrong chunks, stage 2 at unsupported claims inside the right chunk (README, results 3-5).

Trust = fraction of sentences that end up SUPPORTED or REATTRIBUTED.

Scorers are swappable: LexicalScorer (our IR pipeline) or NLIScorer (cross-encoder entailment).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from src.index import InvertedIndex
from src.sparse import bm25_idf, bm25_term
from src.text import STOPWORDS, _stem

SUPPORTED, REATTRIBUTED, UNSUPPORTED = "SUPPORTED", "REATTRIBUTED", "UNSUPPORTED"
UNVERIFIED = "UNVERIFIED"


@dataclass
class Support:
    score: float
    cosine: float = 0.0
    coverage: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)   # shared term → cosine contribution
    missing: list[str] = field(default_factory=list)        # numbers/entities absent from the chunk


# ---------- anchors: numbers and named entities a claim must not invent ----------
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())} | {
    "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100"}
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# In a claim, a number word counts only when it quantifies ("three-month", "two years"),
# so the pronoun "one" ("one should…") is not an anchor.
CLAIM_NUMWORD_RE = re.compile(
    r"\b(" + "|".join(NUMBER_WORDS) + r")(?=[-\u2010\u2011 ](?:year|month|week|day|percent|time|hundred|thousand|million|billion)s?\b|[-\u2010\u2011])",
    re.I)
ANY_NUMWORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019]*")


def _norm_number(s: str) -> str:
    s = s.rstrip(",").replace(",", "")
    return s[:-2] if s.endswith(".0") else s


def _norm_word(w: str) -> str:
    w = re.sub(r"['\u2019]s?$", "", w)
    return w.lower() if w.isupper() else _stem(w.lower())


def _fractions(text: str) -> str:
    return text.replace("\u00bd", ".5").replace("\u00bc", ".25").replace("\u00be", ".75")


def anchors(sentence: str) -> dict[str, set[str]]:
    """Numbers (digits or quantifying number words) and named entities (capitalised words that
    are not sentence-initial, or ALL-CAPS acronyms anywhere), normalised for matching."""
    sentence = _fractions(sentence)
    numbers = {_norm_number(m) for m in NUM_RE.findall(sentence)}
    numbers |= {NUMBER_WORDS[m.lower()] for m in CLAIM_NUMWORD_RE.findall(sentence)}
    entities = set()
    for i, m in enumerate(WORD_RE.finditer(sentence)):
        w = re.sub(r"['\u2019]s?$", "", m.group())
        if len(w) < 2 or w.lower() in STOPWORDS or not w[0].isupper():
            continue
        if i == 0 and not w.isupper():
            continue
        entities.add(_norm_word(w))
    return {"numbers": numbers, "entities": entities}


def chunk_vocabulary(chunk: str) -> tuple[set[str], set[str]]:
    """Everything a chunk can vouch for: its numbers (digits and any number word) and all its
    words in both acronym (lower) and stemmed form."""
    chunk = _fractions(chunk)
    numbers = {_norm_number(m) for m in NUM_RE.findall(chunk)}
    numbers |= {NUMBER_WORDS[m.lower()] for m in ANY_NUMWORD_RE.findall(chunk)}
    words = set()
    for m in WORD_RE.finditer(chunk):
        w = re.sub(r"['\u2019]s?$", "", m.group()).lower()
        words |= {w, _stem(w)}
    return numbers, words


def missing_anchors(sentence: str, chunk: str) -> tuple[list[str], int]:
    """(anchors of the sentence absent from the chunk, total anchors in the sentence)."""
    a = anchors(sentence)
    numbers, words = chunk_vocabulary(chunk)
    missing = sorted(a["numbers"] - numbers) + sorted(a["entities"] - words)
    return missing, len(a["numbers"]) + len(a["entities"])


@dataclass
class Verdict:
    sentence: str
    cited: int | None          # 1-based chunk number the LLM cited
    final: int | None          # chunk number after verification (None if unsupported)
    status: str
    score: float               # support of the final (or best-tried) chunk
    cited_score: float         # support of the originally cited chunk
    terms: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    nli: float | None = None   # stage-2 entailment of the final chunk (None if stage 2 did not run)


@dataclass
class Report:
    sentences: list[Verdict]

    @property
    def trust(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(v.status in (SUPPORTED, REATTRIBUTED) for v in self.sentences) / len(self.sentences)


class LexicalScorer:
    name = "lexical"

    def __init__(self, index: InvertedIndex, alpha: float = 0.5, beta: float = 0.0):
        self.index, self.alpha, self.beta = index, alpha, beta
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
        base = self.alpha * cosine + (1 - self.alpha) * coverage
        missing, total = missing_anchors(sentence, chunk) if self.beta else ([], 0)
        penalty = 1 - self.beta * (len(missing) / total) if total else 1.0
        return Support(base * penalty, cosine, coverage, contrib, missing)


class NLIScorer:
    """Entailment probability P(chunk ⊨ sentence) from an NLI cross-encoder.
    This is a model judgment call by design: it is the learned baseline CiteGuard is compared to."""
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)   # FiQA chunks can exceed the 4 GB GPU at full length
        labels = self.model.model.config.id2label
        self.entail = next(i for i, l in labels.items() if l.lower().startswith("entail"))

    def scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch: pairs of (sentence, chunk) → entailment probabilities."""
        if not pairs:
            return []
        probs = self.model.predict([(c, s) for s, c in pairs], apply_softmax=True, batch_size=8, show_progress_bar=False)
        return [float(p[self.entail]) for p in probs]

    def score(self, sentence: str, chunk: str) -> Support:
        return Support(self.scores([(sentence, chunk)])[0])


class CiteGuard:
    def __init__(self, scorer, index: InvertedIndex, threshold: float, k1: float = 1.2, b: float = 0.75,
                 verifier=None, verify_threshold: float = 0.5):
        self.scorer, self.index, self.threshold = scorer, index, threshold
        self.k1, self.b = k1, b
        self.verifier, self.verify_threshold = verifier, verify_threshold   # stage 2 (NLIScorer-like)

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
                verdicts.append(Verdict(sentence, cited, cited, SUPPORTED, sup.score, sup.score, sup.terms, sup.missing))
                continue
            alt = self.best_alternative(sentence, chunks, exclude=cited if valid else None)
            alt_sup = self.scorer.score(sentence, chunks[alt - 1]) if alt else Support(0.0)
            if alt and alt_sup.score >= self.threshold:
                verdicts.append(Verdict(sentence, cited, alt, REATTRIBUTED, alt_sup.score, sup.score, alt_sup.terms,
                                        alt_sup.missing))
            else:
                verdicts.append(Verdict(sentence, cited, None, UNSUPPORTED, sup.score, sup.score, sup.terms, sup.missing))
        if self.verifier is not None:
            self._verify(verdicts, chunks)
        return Report(verdicts)

    def _verify(self, verdicts: list[Verdict], chunks: list[str]) -> None:
        """Stage 2: one batched NLI call over every sentence that passed stage 1."""
        todo = [v for v in verdicts if v.status in (SUPPORTED, REATTRIBUTED)]
        probs = self.verifier.scores([(v.sentence, chunks[v.final - 1]) for v in todo])
        for v, p in zip(todo, probs):
            v.nli = p
            if p < self.verify_threshold:
                v.status = UNVERIFIED
