"""citeguard CLI.

    python -m src.cli ask "Is a Roth IRA better than a 401k?" --retriever bm25 --show-postings --show-scores
    python -m src.cli postings "dividend tax"
"""
from __future__ import annotations

import argparse
import textwrap

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from src.citeguard import REATTRIBUTED, SUPPORTED
from src.retrievers import ALL

console = Console()
COLOURS = {SUPPORTED: "green", REATTRIBUTED: "yellow", "UNSUPPORTED": "red"}


def show_hits(ans, show_scores: bool) -> None:
    t = Table(title=f"retrieved chunks — {ans.question!r}", show_lines=True)
    t.add_column("#", justify="right")
    t.add_column("doc")
    t.add_column("score", justify="right")
    t.add_column("text")
    if show_scores:
        t.add_column("per-term contribution")
    for i, (h, chunk) in enumerate(zip(ans.hits, ans.chunks), 1):
        row = [str(i), h.doc_id, f"{h.score:.4f}", escape(textwrap.shorten(chunk, 220))]
        if show_scores:
            row.append(", ".join(f"{t}={s:.3f}" for t, s in sorted(h.terms.items(), key=lambda x: -x[1])) or "—")
        t.add_row(*row)
    console.print(t)


def show_verdicts(ans) -> None:
    if not ans.pairs:
        console.print(Panel(escape(ans.raw.strip()), title="answer (no cited claims)", border_style="red"))
        return
    console.print(Panel(escape(ans.raw.strip()), title="raw LLM answer"))
    t = Table(title="CiteGuard verdicts", show_lines=True)
    for col in ("sentence", "cite", "status", "support", "contributing terms"):
        t.add_column(col)
    for v in ans.report.sentences:
        c = COLOURS[v.status]
        cite = f"[{v.cited}]" if v.status != REATTRIBUTED else f"[{v.cited}] → [{v.final}]"
        terms = ", ".join(f"{k}" for k, _ in sorted(v.terms.items(), key=lambda x: -x[1])[:6]) or "—"
        score = f"{v.score:.3f}" if v.status != REATTRIBUTED else f"{v.cited_score:.3f} → {v.score:.3f}"
        t.add_row(escape(v.sentence), escape(cite), f"[{c}]{v.status}[/]", score, terms)
    console.print(t)
    trust = ans.report.trust
    colour = "green" if trust >= 0.8 else "yellow" if trust >= 0.5 else "red"
    console.print(f"[bold {colour}]trust score: {trust:.2f}[/]  "
                  f"({sum(v.status != 'UNSUPPORTED' for v in ans.report.sentences)}/{len(ans.report.sentences)} sentences supported)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="retrieve, answer with citations, verify")
    a.add_argument("question")
    a.add_argument("--retriever", default="hybrid", choices=ALL)
    a.add_argument("--full", action="store_true", help="use the full corpus (default: 5k sample)")
    a.add_argument("--nli", action="store_true", help="verify with the NLI cross-encoder instead of lexical support")
    a.add_argument("--show-postings", action="store_true")
    a.add_argument("--show-scores", action="store_true")
    p = sub.add_parser("postings", help="pretty-print postings and df/idf for query words")
    p.add_argument("words")
    p.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.cmd == "postings":
        from src.build import get_index
        from src.config import load_config

        idx = get_index(load_config(), args.full)
        for w in args.words.split():
            idx.describe_term(w, console=console)
        return

    from src.pipeline import RAG

    with console.status("loading index and models …"):
        rag = RAG(full=args.full, retriever=args.retriever, nli=args.nli)
    if args.show_postings:
        for w in dict.fromkeys(args.question.split()):
            if rag.index.processor.process(w):
                rag.index.describe_term(w, limit=6, console=console)
    with console.status(f"retrieving ({args.retriever}) and generating …"):
        ans = rag.ask(args.question, explain=args.show_scores)
    show_hits(ans, args.show_scores)
    show_verdicts(ans)


if __name__ == "__main__":
    main()
