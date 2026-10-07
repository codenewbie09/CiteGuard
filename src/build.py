"""Build (or load cached) indexes.  `python -m src.build [--full] [--term WORD ...]`."""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from src.config import load_config, path
from src.data import corpus_docs
from src.index import InvertedIndex
from src.text import TextProcessor


def index_path(cfg: dict, full: bool, proc: TextProcessor):
    flags = "".join("1" if x else "0" for x in proc.settings)  # lowercase, stopwords, stem
    return path(cfg["index"]["dir"]) / f"{'full' if full else 'sample'}_{flags}.pkl"


def get_index(cfg: dict, full: bool = False, proc: TextProcessor | None = None,
              rebuild: bool = False, verbose: bool = True) -> InvertedIndex:
    proc = proc or TextProcessor.from_config(cfg)
    p = index_path(cfg, full, proc)
    if p.exists() and not rebuild:
        return InvertedIndex.load(p)
    docs = corpus_docs(cfg, sample=not full)
    if verbose:
        print(f"building index over {len(docs):,} docs ({p.name}) …")
    idx = InvertedIndex.build(docs, proc, champion_r=cfg["index"]["champion_r"])
    idx.save(p)
    return idx


def print_stats(idx: InvertedIndex, console: Console) -> None:
    t = Table(title="index stats")
    t.add_column("stat")
    t.add_column("value", justify="right")
    for k, v in idx.stats().items():
        t.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    console.print(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="index the full corpus (default: 5k sample)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--term", nargs="*", default=[], help="pretty-print postings for these words")
    args = ap.parse_args()
    cfg = load_config()
    console = Console()
    idx = get_index(cfg, args.full, rebuild=args.rebuild)
    print_stats(idx, console)
    for w in args.term:
        idx.describe_term(w, console=console)


if __name__ == "__main__":
    main()
