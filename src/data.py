"""BEIR FiQA loading. Source: Thakur et al., BEIR (2021); FiQA-2018 (Maia et al., 2018).

What is a document: one FiQA corpus passage (a StackExchange answer) = one document.
FiQA passages have no titles (all 57,638 are empty), so the "title" zone is the passage's
lead sentence — see docs/adr/0002-lead-sentence-title-zone.md.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
import zipfile
from functools import lru_cache

from src.config import path

LEAD_RE = re.compile(r"(?<=[.!?])\s+")
LEAD_MAX_WORDS = 30


def ensure_fiqa(cfg: dict):
    root = path(cfg["data"]["dir"])
    target = root / "fiqa"
    if not (target / "corpus.jsonl").exists():
        root.mkdir(parents=True, exist_ok=True)
        zpath = root / "fiqa.zip"
        if not zpath.exists():
            print(f"downloading {cfg['data']['url']} …")
            urllib.request.urlretrieve(cfg["data"]["url"], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    return target


def lead_sentence(text: str) -> str:
    first = LEAD_RE.split(text.strip(), maxsplit=1)[0]
    return " ".join(first.split()[:LEAD_MAX_WORDS])


@lru_cache(maxsize=2)
def _corpus(root: str) -> dict[str, dict]:
    docs = {}
    with open(f"{root}/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            body = d["text"]
            title = d["title"].strip() or lead_sentence(body)
            docs[d["_id"]] = {"id": d["_id"], "title": title, "body": body}
    return docs


def load_corpus(cfg: dict) -> dict[str, dict]:
    return _corpus(str(ensure_fiqa(cfg)))


def load_queries(cfg: dict) -> dict[str, str]:
    with open(ensure_fiqa(cfg) / "queries.jsonl") as f:
        return {q["_id"]: q["text"] for q in map(json.loads, f)}


def load_qrels(cfg: dict, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(ensure_fiqa(cfg) / "qrels" / f"{split}.tsv") as f:
        next(f)
        for line in f:
            qid, did, grade = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(grade)
    return qrels


def query_sample(cfg: dict, split: str, n: int) -> list[str]:
    """A fixed, seeded sample of query ids that have qrels in `split`."""
    qids = sorted(load_qrels(cfg, split), key=int)
    rng = random.Random(cfg["seed"])
    return sorted(rng.sample(qids, min(n, len(qids))), key=int)


def corpus_docs(cfg: dict, sample: bool) -> list[dict]:
    """Full corpus, or the sample-mode subset: every judged doc for the eval/dev queries plus
    random filler up to `sample_size`. Sample-mode metrics are therefore optimistic (fewer
    distractors); headline numbers come from the full corpus."""
    corpus = load_corpus(cfg)
    if not sample:
        return list(corpus.values())
    keep = set()
    for split, n in (("test", cfg["data"]["n_eval_queries"]), ("dev", cfg["data"]["n_dev_queries"])):
        qrels = load_qrels(cfg, split)
        for qid in query_sample(cfg, split, n):
            keep.update(qrels[qid])
    rest = sorted(set(corpus) - keep, key=int)
    rng = random.Random(cfg["seed"])
    keep.update(rng.sample(rest, max(0, cfg["data"]["sample_size"] - len(keep))))
    return [corpus[d] for d in sorted(keep, key=int)]
