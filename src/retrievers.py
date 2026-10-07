"""One place that turns a retriever name into a configured retriever."""
from __future__ import annotations

from src.index import InvertedIndex
from src.sparse import BM25Retriever, TfidfRetriever

SPARSE = ["tfidf", "tfidf+zones", "tfidf+champions", "bm25"]
ALL = SPARSE + ["dense", "hybrid", "hybrid-equal"]


def make_retriever(name: str, cfg: dict, index: InvertedIndex, full: bool = False):
    r = cfg["retrieval"]
    if name == "tfidf":
        return TfidfRetriever(index)
    if name == "tfidf+zones":
        return TfidfRetriever(index, zones=True, w_title=r["w_title"], w_body=r["w_body"])
    if name == "tfidf+champions":
        return TfidfRetriever(index, champion_r=cfg["index"]["champion_r"])
    if name == "bm25":
        return BM25Retriever(index, k1=r["bm25_k1"], b=r["bm25_b"])
    if name in ("dense", "hybrid", "hybrid-equal"):
        from src.dense import DenseRetriever, HybridRetriever

        dense = DenseRetriever.for_index(cfg, index, full)
        if name == "dense":
            return dense
        bm25 = BM25Retriever(index, r["bm25_k1"], r["bm25_b"])
        if name == "hybrid-equal":                   # plain RRF, k=60, for comparison
            return HybridRetriever([bm25, dense], k=60, name="hybrid-equal")
        w = r["rrf_w_dense"]
        return HybridRetriever([bm25, dense], k=r["rrf_k"], weights=[1 - w, w])
    raise ValueError(f"unknown retriever {name!r}; choose from {ALL}")
