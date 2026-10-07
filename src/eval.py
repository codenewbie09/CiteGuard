"""Module 7 — evaluation.  Results go to results/ as CSV + PNG.

    python -m src.eval retrievers [--full] [--only bm25 tfidf ...]
    python -m src.eval ablation   [--full]
    python -m src.eval zones      [--full]     # tune w_title on the dev split
    python -m src.eval rrf        [--full]     # tune weighted-RRF w_dense and k on the dev split
    python -m src.eval citeguard  [--full]     # corruption test; tunes alpha/threshold on dev
    python -m src.eval labels                  # dump answers for manual labelling
    python -m src.eval agreement               # agreement once labels are filled in
"""
from __future__ import annotations

import argparse
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.build import get_index
from src.citeguard import SUPPORTED, CiteGuard, LexicalScorer, NLIScorer
from src.config import load_config, path
from src.data import load_qrels, load_queries, query_sample
from src.metrics import mrr_at_k, ndcg_at_k, paired_randomization, precision_at_k, recall_at_k
from src.retrievers import ALL, SPARSE, make_retriever
from src.text import TextProcessor

console = Console()
METRICS = ["P@5", "P@10", "Recall@100", "nDCG@10", "MRR@10"]


def results_dir():
    d = path("results")
    d.mkdir(exist_ok=True)
    return d


def evaluate(retriever, queries: dict[str, str], qrels: dict, depth: int = 100, per_query: list | None = None) -> dict:
    """Mean metrics and mean latency (ms) of one retriever over the query set.
    If `per_query` is a list, the per-query metric rows are appended to it."""
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
    if per_query is not None:
        per_query.extend(rows)
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


def run_retrievers(cfg, full: bool, names: list[str], n_queries: int | None = None) -> pd.DataFrame:
    queries, qrels = eval_queries(cfg)
    idx = get_index(cfg, full)
    mode = "full" if full else "sample"
    if n_queries:
        mode += f"_q{len(queries)}"
    rows, per_query = {}, {}
    for name in names:
        console.print(f"[bold]{name}[/] on {len(queries)} test queries ({mode}, N={idx.N:,})")
        per_query[name] = []
        rows[name] = evaluate(make_retriever(name, cfg, idx, full), queries, qrels,
                              cfg["retrieval"]["top_k"], per_query[name])
    df = pd.DataFrame(rows).T
    df.index.name = "retriever"
    df.to_csv(results_dir() / f"retrievers_{mode}.csv", float_format="%.4f")
    bar_chart(df, f"FiQA test ({len(queries)} queries, {mode} corpus, N={idx.N:,})",
              results_dir() / f"retrievers_{mode}.png")
    show(df, f"retriever comparison — {mode}")
    significance(per_query, mode)
    return df


def significance(per_query: dict[str, list[dict]], mode: str) -> None:
    """Paired randomisation tests of each hybrid against its best component (dense)."""
    pairs = [(h, "dense") for h in ("hybrid", "hybrid-equal") if h in per_query and "dense" in per_query]
    if not pairs:
        return
    rows = []
    for a, b in pairs:
        for m in ("nDCG@10", "MRR@10", "P@10", "Recall@100"):
            x, y = [r[m] for r in per_query[a]], [r[m] for r in per_query[b]]
            d = [i - j for i, j in zip(x, y)]
            rows.append({"comparison": f"{a} vs {b}", "metric": m, "mean_diff": sum(d) / len(d),
                         "wins": sum(v > 0 for v in d), "ties": sum(v == 0 for v in d),
                         "losses": sum(v < 0 for v in d), "p_value": paired_randomization(x, y)})
    df = pd.DataFrame(rows).set_index("comparison")
    df.to_csv(results_dir() / f"significance_{mode}.csv", float_format="%.4f")
    show(df, "paired randomisation tests (two-sided, 10k sign flips)")


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


def run_zones(cfg, full: bool) -> pd.DataFrame:
    """Tune the title-zone weight on dev queries (never test)."""
    from src.sparse import TfidfRetriever

    queries, qrels = eval_queries(cfg, "dev")
    idx = get_index(cfg, full)
    rows = {}
    for wt in cfg["experiment"]["w_title_grid"]:
        r = TfidfRetriever(idx, zones=True, w_title=wt, w_body=1 - wt)
        rows[f"w_title={wt}"] = evaluate(r, queries, qrels, cfg["retrieval"]["top_k"])
    df = pd.DataFrame(rows).T
    df.index.name = "zone weights (dev)"
    df.to_csv(results_dir() / f"zones_dev_{'full' if full else 'sample'}.csv", float_format="%.4f")
    show(df, "zone-weight tuning on dev")
    return df


def run_rrf(cfg, full: bool) -> pd.DataFrame:
    """Tune weighted RRF (w_dense, k) on dev queries. Each base run is retrieved once and
    fused in memory for every grid point."""
    from src.dense import HybridRetriever

    queries, qrels = eval_queries(cfg, "dev")
    idx = get_index(cfg, full)
    depth = cfg["retrieval"]["top_k"]
    bases = [make_retriever(n, cfg, idx, full) for n in ("bm25", "dense")]
    runs = {q: [[h.doc_id for h in r.search(t, k=depth)] for r in bases] for q, t in queries.items()}
    rows = []
    for k in cfg["experiment"]["rrf_k_grid"]:
        for w in cfg["experiment"]["rrf_w_dense_grid"]:
            ranked = {q: [d for d, _ in HybridRetriever.fuse(runs[q], k, [1 - w, w])[:depth]] for q in queries}
            m = {name: sum(f(ranked[q], qrels[q], kk) for q in queries) / len(queries)
                 for name, f, kk in [("nDCG@10", ndcg_at_k, 10), ("MRR@10", mrr_at_k, 10),
                                     ("Recall@100", recall_at_k, 100), ("P@10", precision_at_k, 10)]}
            rows.append({"k": k, "w_dense": w, **m})
    for name, i in (("bm25 alone", 0), ("dense alone", 1)):
        rel = [(runs[q][i], qrels[q]) for q in queries]
        rows.append({"k": None, "w_dense": name, "nDCG@10": sum(ndcg_at_k(r, g, 10) for r, g in rel) / len(rel),
                     "MRR@10": sum(mrr_at_k(r, g, 10) for r, g in rel) / len(rel),
                     "Recall@100": sum(recall_at_k(r, g, 100) for r, g in rel) / len(rel),
                     "P@10": sum(precision_at_k(r, g, 10) for r, g in rel) / len(rel)})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / f"rrf_dev_{'full' if full else 'sample'}.csv", index=False, float_format="%.4f")
    grid = df[df.k.notna()]
    best = grid.loc[grid["nDCG@10"].idxmax()]
    show(df.set_index("w_dense"), "weighted RRF tuning on dev")
    console.print(f"[bold]best on dev:[/] k={int(best.k)}, w_dense={best.w_dense} (nDCG@10 {best['nDCG@10']:.4f})")
    return df


# ---------------- CiteGuard corruption experiment ----------------

def corrupt(pairs, n_chunks: int, rate: float, rng: random.Random) -> list[dict]:
    """Swap each valid citation to a different retrieved chunk with probability `rate`.
    The original citation is kept as ground truth."""
    out = []
    for sentence, cited in pairs:
        item = {"sentence": sentence, "original": cited, "cited": cited, "corrupted": False}
        if cited is not None and 1 <= cited <= n_chunks and rng.random() < rate:
            item["cited"] = rng.choice([c for c in range(1, n_chunks + 1) if c != cited])
            item["corrupted"] = True
        out.append(item)
    return out


def prf(pred: list[bool], truth: list[bool]) -> tuple[float, float, float]:
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(t and not p for p, t in zip(pred, truth))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def generate_answers(cfg, split: str, n: int, full: bool, rag=None) -> list[dict]:
    """Generate (or load cached) answers for n queries of `split`. Cached in results/ so the
    experiment reproduces from a fresh clone without an API key."""
    cache = results_dir() / f"answers_{split}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else []
    if len(cached) >= n:
        return cached[:n]
    from src.generate import parse_answer
    from src.pipeline import RAG

    rag = rag or RAG(cfg, full=full, retriever=cfg["experiment"]["retriever"])
    queries = load_queries(cfg)
    done = {a["qid"] for a in cached}
    for qid in query_sample(cfg, split, max(n, cfg["data"]["n_dev_queries"])):
        if len(cached) >= n:
            break
        if qid in done:
            continue
        hits = rag.retrieve(queries[qid])
        chunks = [rag.corpus[h.doc_id]["body"] for h in hits]
        raw = rag.generate(queries[qid], chunks)
        cached.append({"qid": qid, "question": queries[qid], "doc_ids": [h.doc_id for h in hits],
                       "chunks": chunks, "raw": raw, "pairs": parse_answer(raw)})
        console.print(f"  {split} {len(cached)}/{n}: {len(cached[-1]['pairs'])} sentences — {queries[qid][:60]}")
        cache.write_text(json.dumps(cached, indent=1))
    return cached


def corrupted_items(cfg, answers) -> list[dict]:
    rng = random.Random(cfg["seed"])
    items = []
    for a in answers:
        for it in corrupt(a["pairs"], len(a["chunks"]), cfg["citeguard"]["corruption_rate"], rng):
            if it["original"] is not None:          # uncited sentences have no ground truth
                items.append({**it, "qid": a["qid"], "chunks": a["chunks"]})
    return items


def support_scores(scorer, items) -> list[float]:
    pairs = [(it["sentence"], it["chunks"][it["cited"] - 1]) for it in items]
    if hasattr(scorer, "scores"):
        return scorer.scores(pairs)
    return [scorer.score(s, c).score for s, c in pairs]


THRESHOLDS = [round(t * 0.01, 2) for t in range(0, 101)]


def sweep(scores, truth) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        p, r, f = prf([s < t for s in scores], truth)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def best_threshold(scores, truth) -> float:
    curve = sweep(scores, truth)
    return float(curve.loc[curve.f1.idxmax(), "threshold"])


def recovery(guard: CiteGuard, items) -> dict:
    """Run full CiteGuard on corrupted citations; how often is the original chunk recovered?"""
    flagged = recovered = 0
    corrupted = [it for it in items if it["corrupted"]]
    for it in corrupted:
        v = guard.check([(it["sentence"], it["cited"])], it["chunks"]).sentences[0]
        if v.status != SUPPORTED:
            flagged += 1
            recovered += v.final == it["original"]
    return {"corrupted": len(corrupted), "flagged": flagged, "recovered_original": recovered,
            "recovery_rate_of_flagged": recovered / flagged if flagged else 0.0,
            "recovery_rate_of_corrupted": recovered / len(corrupted) if corrupted else 0.0}


def run_citeguard(cfg, full: bool, nli: bool = True) -> pd.DataFrame:
    e, c = cfg["experiment"], cfg["citeguard"]
    from src.pipeline import RAG

    rag = None
    if not (results_dir() / "answers_test.json").exists() or not (results_dir() / "answers_dev.json").exists():
        rag = RAG(cfg, full=full, retriever=e["retriever"])
    dev = corrupted_items(cfg, generate_answers(cfg, "dev", e["n_dev_answers"], full, rag))
    test = corrupted_items(cfg, generate_answers(cfg, "test", c["n_corruption_queries"], full, rag))
    dev_truth, test_truth = [i["corrupted"] for i in dev], [i["corrupted"] for i in test]
    console.print(f"dev: {len(dev)} sentences ({sum(dev_truth)} corrupted); "
                  f"test: {len(test)} sentences ({sum(test_truth)} corrupted)")
    idx = get_index(cfg, full)

    # 1. tune alpha, beta and threshold on dev (beta = 0 is the plain lexical checker)
    grid = []
    for b in e["betas"]:
        for a in e["alphas"]:
            sc = support_scores(LexicalScorer(idx, a, b), dev)
            t = best_threshold(sc, dev_truth)
            grid.append({"alpha": a, "beta": b, "best_threshold": t, "dev_f1": prf([s < t for s in sc], dev_truth)[2]})
    alpha_df = pd.DataFrame(grid)
    alpha_df.to_csv(results_dir() / "citeguard_alpha_dev.csv", index=False, float_format="%.4f")
    plain = alpha_df[alpha_df.beta == 0].sort_values("dev_f1", ascending=False).iloc[0]
    anch = alpha_df[alpha_df.beta > 0].sort_values("dev_f1", ascending=False).iloc[0]
    checkers = {
        "lexical": (LexicalScorer(idx, plain.alpha, 0.0), float(plain.best_threshold)),
        "lexical+anchors": (LexicalScorer(idx, anch.alpha, anch.beta), float(anch.best_threshold)),
    }
    if nli:
        nli_scorer = NLIScorer(c["nli_model"])
        checkers["nli"] = (nli_scorer, best_threshold(support_scores(nli_scorer, dev), dev_truth))
    tuned = {n: {"alpha": getattr(sc, "alpha", None), "beta": getattr(sc, "beta", None), "threshold": t}
             for n, (sc, t) in checkers.items()}
    (results_dir() / "citeguard_tuned.json").write_text(json.dumps(tuned, indent=1))

    # 2. evaluate on test with the dev-tuned settings
    rows, curves, per_sentence = [], {}, pd.DataFrame(
        {"qid": [i["qid"] for i in test], "sentence": [i["sentence"] for i in test],
         "original": [i["original"] for i in test], "cited": [i["cited"] for i in test], "corrupted": test_truth})
    for name, (scorer, t) in checkers.items():
        sc = support_scores(scorer, test)
        per_sentence[f"{name}_support"] = sc
        p, r, f = prf([s < t for s in sc], test_truth)
        curves[name] = sweep(sc, test_truth)
        rec = recovery(CiteGuard(scorer, idx, t, cfg["retrieval"]["bm25_k1"], cfg["retrieval"]["bm25_b"]), test)
        rows.append({"checker": name, "alpha": getattr(scorer, "alpha", None), "beta": getattr(scorer, "beta", None),
                     "threshold (dev-tuned)": t,
                     "precision": p, "recall": r, "f1": f, **rec})
    df = pd.DataFrame(rows).set_index("checker")
    df.to_csv(results_dir() / "citeguard_corruption.csv", float_format="%.4f")
    per_sentence.to_csv(results_dir() / "citeguard_sentences_test.csv", index=False, float_format="%.4f")
    pd.concat({k: v.set_index("threshold") for k, v in curves.items()}, axis=1).to_csv(
        results_dir() / "citeguard_threshold_curve.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in curves.items():
        line, = ax.plot(curve.threshold, curve.f1, label=f"{name} checker")
        t = checkers[name][1]
        ax.axvline(t, color=line.get_color(), ls=":", lw=1)
        ax.annotate(f"dev-tuned {t:.2f}", (t, curve.set_index("threshold").f1.get(t, 0)),
                    textcoords="offset points", xytext=(5, 8), color=line.get_color(), fontsize=8)
    ax.set_xlabel("support threshold (flag sentence if support < threshold)")
    ax.set_ylabel("F1 detecting swapped citations (test)")
    ax.set_title(f"CiteGuard corruption test — {sum(test_truth)}/{len(test)} citations swapped")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir() / "citeguard_threshold_curve.png", dpi=150)
    plt.close()
    show(alpha_df.sort_values("dev_f1", ascending=False).head(8).set_index("alpha"),
         "lexical alpha/beta tuning on dev (top 8)")
    show(df, "CiteGuard corruption test (test split, dev-tuned settings)")
    return df


# ---------------- manual labelling ----------------

LABELS = "manual_labels.csv"


def dump_labels(cfg, full: bool, force: bool = False) -> None:
    out = results_dir() / LABELS
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (it may hold your labels); pass --force to overwrite")
    answers = generate_answers(cfg, "test", cfg["experiment"]["n_label_answers"], full)
    idx = get_index(cfg, full)
    c = cfg["citeguard"]
    guard = CiteGuard(LexicalScorer(idx, c["alpha"]), idx, c["threshold"])
    rows = []
    for a in answers:
        for n, v in enumerate(guard.check([tuple(p) for p in a["pairs"]], a["chunks"]).sentences, 1):
            rows.append({"qid": a["qid"], "question": a["question"], "sentence_no": n, "sentence": v.sentence,
                         "cited": v.cited,
                         "cited_chunk": a["chunks"][v.cited - 1] if v.cited and v.cited <= len(a["chunks"]) else "",
                         "citeguard_status": v.status, "support": round(v.cited_score, 4), "final": v.final,
                         "human_label": ""})
    pd.DataFrame(rows).to_csv(out, index=False)
    console.print(f"wrote {len(rows)} sentences to {out}. Fill human_label with 1 (cited chunk supports "
                  "the sentence) or 0 (it does not), then run: python -m src.eval agreement")


def agreement(cfg, nli: bool = True) -> pd.DataFrame:
    """Agreement between human labels and each checker. Each checker is re-run on the labelled
    (sentence, cited chunk) pairs with its dev-tuned settings (results/citeguard_tuned.json),
    so the labels are a held-out test and are never used for tuning."""
    df = pd.read_csv(results_dir() / LABELS)
    df = df[df.human_label.notna() & (df.human_label.astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit("no human labels filled in yet")
    human = df.human_label.astype(int).astype(bool).tolist()
    tuned_path = results_dir() / "citeguard_tuned.json"
    c = cfg["citeguard"]
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {
        "lexical": {"alpha": c["alpha"], "beta": c.get("beta", 0.0), "threshold": c["threshold"]}}
    idx = get_index(cfg, True)
    pairs = [(s, ch if isinstance(ch, str) else "") for s, ch in zip(df.sentence, df.cited_chunk)]
    rows, verdicts = [], {}
    for name, p in tuned.items():
        if name == "nli":
            if not nli:
                continue
            scores = NLIScorer(c["nli_model"]).scores(pairs)
        else:
            scorer = LexicalScorer(idx, p["alpha"], p["beta"])
            scores = [scorer.score(s, ch).score for s, ch in pairs]
        machine = [x >= p["threshold"] for x in scores]
        verdicts[name] = machine
        n = len(human)
        po = sum(h == m for h, m in zip(human, machine)) / n
        ph, pm = sum(human) / n, sum(machine) / n
        pe = ph * pm + (1 - ph) * (1 - pm)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        # "positive" = unsupported, the thing CiteGuard is meant to catch
        pr, rc, f = prf([not m for m in machine], [not h for h in human])
        rows.append({"checker": name, **{k: v for k, v in p.items()}, "labelled": n, "accuracy": po,
                     "cohen_kappa": kappa, "unsupported_precision": pr, "unsupported_recall": rc,
                     "unsupported_f1": f, "flagged": sum(not m for m in machine), "human_unsupported": n - sum(human)})
    out = pd.DataFrame(rows).set_index("checker")
    out.to_csv(results_dir() / "manual_agreement.csv", float_format="%.4f")
    detail = df[["qid", "sentence", "human_label"]].copy()
    for name, m in verdicts.items():
        detail[f"{name}_supported"] = [int(x) for x in m]
    detail.to_csv(results_dir() / "manual_agreement_detail.csv", index=False)
    show(out, "agreement with human labels (held-out; settings tuned on dev corruption set)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["retrievers", "ablation", "zones", "rrf", "citeguard", "labels", "agreement"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--n-queries", type=int, default=None,
                    help="retrievers: evaluate on this many test queries (outputs get a _q<N> suffix)")
    ap.add_argument("--no-nli", action="store_true", help="citeguard: skip the NLI checker")
    ap.add_argument("--force", action="store_true", help="labels: overwrite an existing labels file")
    args = ap.parse_args()
    cfg = load_config()
    if args.experiment == "retrievers":
        if args.n_queries:   # in-memory only: the sample corpus is still built from the configured count
            cfg["data"]["n_eval_queries"] = args.n_queries
        run_retrievers(cfg, args.full, args.only or SPARSE, args.n_queries)
    elif args.experiment == "ablation":
        run_ablation(cfg, args.full)
    elif args.experiment == "zones":
        run_zones(cfg, args.full)
    elif args.experiment == "rrf":
        run_rrf(cfg, args.full)
    elif args.experiment == "citeguard":
        run_citeguard(cfg, args.full, nli=not args.no_nli)
    elif args.experiment == "labels":
        dump_labels(cfg, args.full, args.force)
    elif args.experiment == "agreement":
        agreement(cfg)


if __name__ == "__main__":
    main()
