"""Module 4 — dense retrieval (MiniLM + FAISS inner product) and Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import path
from src.sparse import Hit


class DenseRetriever:
    name = "dense"

    def __init__(self, model, faiss_index, doc_ids: list[str]):
        self.model, self.faiss, self.doc_ids = model, faiss_index, doc_ids

    @classmethod
    def for_index(cls, cfg: dict, inv_index, full: bool) -> DenseRetriever:
        """Embed the same documents as the inverted index (same order); cache vectors to disk."""
        import faiss
        from sentence_transformers import SentenceTransformer

        from src.data import load_corpus

        d = cfg["dense"]
        model = SentenceTransformer(d["model"])
        cache = path(f"data/dense/{'full' if full else 'sample'}.npy")
        ids = inv_index.doc_ids
        if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(ids):
            emb = np.load(cache)
        else:
            corpus = load_corpus(cfg)
            print(f"embedding {len(ids):,} docs with {d['model']} …")
            emb = model.encode([corpus[i]["body"] for i in ids], batch_size=d["batch_size"],
                               normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb.astype(np.float32))
        index = faiss.IndexFlatIP(emb.shape[1])   # normalised vectors: inner product = cosine
        index.add(np.ascontiguousarray(emb, dtype=np.float32))
        return cls(model, index, ids)

    def search(self, query: str, k: int = 10, explain: bool = False) -> list[Hit]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idx = self.faiss.search(q.astype(np.float32), k)
        return [Hit(self.doc_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = Σ_r 1 / (k + rank_r(d)), ranks starting at 1."""
    name = "hybrid"

    def __init__(self, retrievers, k: int = 60, depth: int = 100):
        self.retrievers, self.k, self.depth = retrievers, k, depth

    def search(self, query: str, k: int = 10, explain: bool = False) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        parts: dict[str, dict[str, float]] = defaultdict(dict)
        for r in self.retrievers:
            for rank, h in enumerate(r.search(query, k=max(k, self.depth)), 1):
                s = 1 / (self.k + rank)
                fused[h.doc_id] += s
                parts[h.doc_id][r.name] = s
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [Hit(d, s, parts[d] if explain else {}) for d, s in ranked]
