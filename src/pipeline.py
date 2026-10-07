"""retrieve → generate → verify, as one object shared by the CLI, the app and the evals."""
from __future__ import annotations

from dataclasses import dataclass

from src.build import get_index
from src.citeguard import CiteGuard, LexicalScorer, NLIScorer, Report
from src.config import load_config
from src.data import load_corpus
from src.generate import build_prompt, complete, parse_answer
from src.retrievers import make_retriever
from src.sparse import Hit


@dataclass
class Answer:
    question: str
    hits: list[Hit]            # all retrieved hits (chunks = first n_chunks)
    chunks: list[str]
    raw: str
    pairs: list[tuple[str, int | None]]
    report: Report


class RAG:
    def __init__(self, cfg: dict | None = None, full: bool = False, retriever: str = "bm25", nli: bool = False,
                 two_stage: bool = False):
        self.cfg = cfg or load_config()
        self.full = full
        self.index = get_index(self.cfg, full)
        self.corpus = load_corpus(self.cfg)
        self.retriever = make_retriever(retriever, self.cfg, self.index, full)
        c, r = self.cfg["citeguard"], self.cfg["retrieval"]
        if nli:
            scorer, thr = NLIScorer(c["nli_model"]), c["nli_threshold"]
        else:
            scorer, thr = LexicalScorer(self.index, c["alpha"], c.get("beta", 0.0)), c["threshold"]
        verifier = NLIScorer(c["nli_model"]) if two_stage and not nli else None
        self.guard = CiteGuard(scorer, self.index, thr, r["bm25_k1"], r["bm25_b"],
                               verifier=verifier, verify_threshold=c["verify_threshold"])

    def retrieve(self, question: str, k: int | None = None, explain: bool = False) -> list[Hit]:
        return self.retriever.search(question, k=k or self.cfg["generation"]["n_chunks"], explain=explain)

    def generate(self, question: str, chunks: list[str]) -> str:
        g = self.cfg["generation"]
        return complete(build_prompt(question, chunks), max_tokens=g["max_tokens"], temperature=g["temperature"])

    def ask(self, question: str, explain: bool = False) -> Answer:
        hits = self.retrieve(question, explain=explain)
        chunks = [self.corpus[h.doc_id]["body"] for h in hits]
        raw = self.generate(question, chunks)
        pairs = parse_answer(raw)
        return Answer(question, hits, chunks, raw, pairs, self.guard.check(pairs, chunks))
