"""Module 7 — evaluation.  Results go to results/ as CSV + PNG.

    python -m src.eval retrievers [--full] [--only bm25 tfidf ...]
    python -m src.eval ablation   [--full]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.build import get_index
from src.config import load_config, path
from src.data import load_qrels, load_queries, query_sample
from src.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.retrievers import ALL, SPARSE, make_retriever
from src.text import TextProcessor

console = Console()
METRICS = ["P@5", "P@10", "Recall@100", "nDCG@10", "MRR@10"]


def results_dir():
    d = path("results")
    d.mkdir(exist_ok=True)
    return d


def evaluate(retriever, queries: dict[str, str], qrels: dict, depth: int = 100) -> dict:
    """Mean metrics and mean latency (ms) of one retriever over the query set."""
    rows, lat = [], []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        ranked = [h.doc_id for h in retriever.search(text, k=depth)]
        lat.append((time.perf_counter() - t0) * 1000)
        rel = qrels[qid]
        rows.append({
            "P@5": precision_at_k(ranked, rel, 5), "P@10": precision_at_k(ranked, rel, 10),
            "Recall@100": recall_at_k(ranked, rel, 100), "nDCG@10": ndcg_at_k(ranked, rel, 10),
            "MRR@10": mrr_at_k(ranked, rel, 10),
        })
    out = pd.DataFrame(rows).mean().to_dict()
    out["latency_ms"] = sum(lat) / len(lat)
    return out


def eval_queries(cfg: dict, split: str = "test"):
    n = cfg["data"]["n_eval_queries"] if split == "test" else cfg["data"]["n_dev_queries"]
    qids = query_sample(cfg, split, n)
    queries = load_queries(cfg)
    return {q: queries[q] for q in qids}, load_qrels(cfg, split)


def show(df: pd.DataFrame, title: str) -> None:
    t = Table(title=title)
    t.add_column(df.index.name or "")
    for c in df.columns:
        t.add_column(c, justify="right")
    for name, row in df.iterrows():
        t.add_row(str(name), *[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


def bar_chart(df: pd.DataFrame, title: str, out) -> None:
    ax = df[METRICS].plot.bar(figsize=(10, 5), rot=0, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def run_retrievers(cfg, full: bool, names: list[str]) -> pd.DataFrame:
    queries, qrels = eval_queries(cfg)
    idx = get_index(cfg, full)
    mode = "full" if full else "sample"
    rows = {}
    for name in names:
        console.print(f"[bold]{name}[/] on {len(queries)} test queries ({mode}, N={idx.N:,})")
        rows[name] = evaluate(make_retriever(name, cfg, idx, full), queries, qrels, cfg["retrieval"]["top_k"])
    df = pd.DataFrame(rows).T
    df.index.name = "retriever"
    df.to_csv(results_dir() / f"retrievers_{mode}.csv", float_format="%.4f")
    bar_chart(df, f"FiQA test ({len(queries)} queries, {mode} corpus, N={idx.N:,})",
              results_dir() / f"retrievers_{mode}.png")
    show(df, f"retriever comparison — {mode}")
    return df


def run_ablation(cfg, full: bool) -> pd.DataFrame:
    queries, qrels = eval_queries(cfg)
    mode = "full" if full else "sample"
    rows = []
    for stop in (True, False):
        for stem in (True, False):
            proc = TextProcessor(lowercase=True, remove_stopwords=stop, stem=stem)
            idx = get_index(cfg, full, proc)
            for name in ("tfidf", "bm25"):
                m = evaluate(make_retriever(name, cfg, idx), queries, qrels, cfg["retrieval"]["top_k"])
                rows.append({"retriever": name, "stopwords_removed": stop, "stemming": stem,
                             "vocab": idx.stats()["body_vocab"], **m})
    df = pd.DataFrame(rows).set_index("retriever")
    df.to_csv(results_dir() / f"ablation_{mode}.csv", float_format="%.4f")
    labels = [f"{r}\nstop={'on' if s else 'off'} stem={'on' if t else 'off'}"
              for r, s, t in zip(df.index, df.stopwords_removed, df.stemming)]
    plot = df[METRICS].copy()
    plot.index = labels
    bar_chart(plot.sort_index(), f"Ablation: stemming and stop words ({mode} corpus)",
              results_dir() / f"ablation_{mode}.png")
    show(df, f"ablation — {mode}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["retrievers", "ablation"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    if args.experiment == "retrievers":
        run_retrievers(cfg, args.full, args.only or SPARSE)
    elif args.experiment == "ablation":
        run_ablation(cfg, args.full)


if __name__ == "__main__":
    main()
