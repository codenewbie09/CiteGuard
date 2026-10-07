"""Module 3 — sparse retrievers over the inverted index.

* TfidfRetriever: SMART lnc.ltc vector space model, cosine similarity, term-at-a-time scoring
  with accumulators, heap-based top-K. Optional zone weighting and champion-list fast path.
* BM25Retriever: Okapi BM25 on the body zone of the same index.

Both return a ranked list of Hit(doc_id, score, terms), where `terms` is the per-term
contribution to the score (filled only when explain=True) — the sum of terms equals the score.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from operator import itemgetter

from src.index import InvertedIndex


@dataclass
class Hit:
    doc_id: str
    score: float
    terms: dict[str, float] = field(default_factory=dict)


def top_k(acc: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Heap selection: O(n log k) instead of sorting all accumulators."""
    return heapq.nlargest(k, acc.items(), key=itemgetter(1))


def log_tf(tf: int) -> float:
    return 1 + math.log10(tf) if tf > 0 else 0.0


def ltc_query(index: InvertedIndex, terms: list[str], zone: str) -> dict[str, float]:
    """Query weights: l (1+log10 tf) · t (log10 N/df) · c (cosine-normalised)."""
    w = {t: log_tf(tf) * index.idf(t, zone) for t, tf in Counter(terms).items()}
    w = {t: x for t, x in w.items() if x > 0}
    norm = math.sqrt(sum(x * x for x in w.values()))
    return {t: x / norm for t, x in w.items()} if norm else {}


class TfidfRetriever:
    def __init__(self, index: InvertedIndex, zones: bool = False, w_title: float = 0.3,
                 w_body: float = 0.7, champion_r: int | None = None, name: str | None = None):
        self.index = index
        self.zones = zones
        self.weights = {"title": w_title, "body": w_body} if zones else {"body": 1.0}
        self.champion_r = champion_r
        self.name = name or "tfidf" + ("+zones" if zones else "") + ("+champions" if champion_r else "")

    def _postings(self, term: str, zone: str):
        if self.champion_r and zone == "body":
            return self.index.raw_champions(term, self.champion_r)
        return self.index.raw_postings(term, zone)

    def search(self, query: str, k: int = 10, explain: bool = False) -> list[Hit]:
        idx = self.index
        terms = idx.processor.process(query)
        acc: dict[int, float] = defaultdict(float)
        contrib: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for zone, zw in self.weights.items():
            norms = idx.lnc_norm[zone]
            for term, wq in ltc_query(idx, terms, zone).items():   # term-at-a-time
                docs, tfs = self._postings(term, zone)
                scale = zw * wq
                for d, tf in zip(docs, tfs):
                    # document weight: l (1+log10 tf) · n (no idf) · c (cosine)
                    s = scale * (1 + math.log10(tf)) / norms[d]
                    acc[d] += s
                    if explain:
                        contrib[d][term] += s
        return [Hit(idx.doc_ids[d], s, dict(contrib[d]) if explain else {}) for d, s in top_k(acc, k)]


def bm25_idf(N: int, df: int) -> float:
    """Robertson–Spärck Jones idf with +1 inside the log so it is never negative."""
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def bm25_term(tf: int, dl: int, avgdl: float, idf: float, k1: float, b: float) -> float:
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


class BM25Retriever:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75, name: str = "bm25"):
        self.index, self.k1, self.b, self.name = index, k1, b, name
        self.avgdl = index.avgdl("body")

    def search(self, query: str, k: int = 10, explain: bool = False) -> list[Hit]:
        idx, k1, b, avgdl = self.index, self.k1, self.b, self.avgdl
        dl = idx.doc_len["body"]
        acc: dict[int, float] = defaultdict(float)
        contrib: dict[int, dict[str, float]] = defaultdict(dict)
        for term in dict.fromkeys(idx.processor.process(query)):  # unique, in query order
            docs, tfs = idx.raw_postings(term, "body")
            if not docs:
                continue
            idf = bm25_idf(idx.N, len(docs))
            for d, tf in zip(docs, tfs):
                s = bm25_term(tf, dl[d], avgdl, idf, k1, b)
                acc[d] += s
                if explain:
                    contrib[d][term] = s
        return [Hit(idx.doc_ids[d], s, contrib[d] if explain else {}) for d, s in top_k(acc, k)]
