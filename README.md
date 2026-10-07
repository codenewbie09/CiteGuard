# citeguard

RAG over **BEIR FiQA** (financial QA) with a hand-built, inspectable IR engine and **CiteGuard**,
a verifier that checks every sentence of an LLM answer against the chunk it cites, re-attributes
it to a better chunk when it can, flags it as unsupported when it cannot, and outputs a per-answer
trust score.

IR course hackathon, Track T1 (RAG and trustworthy answers).

```mermaid
flowchart LR
    Q[question] --> TP[text pipeline<br/>src/text.py]
    TP --> SP[sparse: tf-idf lnc.ltc / BM25<br/>src/sparse.py over src/index.py]
    Q --> DE[dense: MiniLM + FAISS<br/>src/dense.py]
    SP --> RRF[hybrid RRF k=60]
    DE --> RRF
    RRF --> TOP[top-5 numbered chunks]
    TOP --> LLM[LLM, one citation per sentence<br/>src/generate.py]
    LLM --> PARSE[(sentence, chunk) pairs]
    PARSE --> CG{CiteGuard<br/>src/citeguard.py<br/>support ≥ τ?}
    CG -- yes --> S[SUPPORTED]
    CG -- no --> BM[BM25 over other chunks]
    BM -- best passes τ --> R[REATTRIBUTED old→new]
    BM -- none passes --> U[UNSUPPORTED]
    S & R & U --> T[trust = supported fraction]
```

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make setup                      # venv + requirements.txt
cp .env.example .env            # set LLM_PROVIDER and the matching API key
make build-index                # downloads FiQA, builds sample + full indexes, prints stats
make test                       # unit tests (toy corpus, hand-verified scores)
```

| step | command |
|---|---|
| ask a question | `python -m src.cli ask "Are Roth IRA withdrawals taxed?" --retriever hybrid --full --show-postings --show-scores` |
| verify with NLI instead | add `--nli` |
| inspect postings / df / idf | `python -m src.cli postings "dividend tax" --full` |
| retriever comparison | `make eval-retrievers` (`FULL=` for the 5k sample) |
| stemming / stop-word ablation | `make eval-ablation` |
| tune zone weights on dev | `make eval-zones` |
| CiteGuard corruption test | `make eval-citeguard` |
| manual labelling | `make labels`, fill `human_label` in `results/manual_labels.csv`, then `make agreement` |
| web page | `make app` (bare Streamlit) |

**Sample vs full mode.** Every command defaults to a 5,000-document sample for fast iteration
(`--full` for the 57,638-document corpus). The sample keeps every judged document for the eval and
dev queries plus seeded random filler, so sample metrics are optimistic; headline numbers below are full-corpus.

**Reproducibility.** Seeds are fixed (`config.yaml: seed`), query samples are seeded, and every
LLM answer used in the experiments is cached in `results/answers_{dev,test}.json`, so
`make eval` reproduces all tables from a fresh clone without an API key.

## Data

[BEIR](https://github.com/beir-cellar/beir) (Thakur et al., NeurIPS 2021 Datasets & Benchmarks) version of
**FiQA-2018** (Maia et al., WWW'18 companion): 57,638 passages, 648 test queries and 500 dev queries with qrels,
downloaded from the BEIR mirror (`config.yaml: data.url`) and cached in `data/`.

**What is a document.** One FiQA corpus passage (a StackExchange answer) = one document = one RAG chunk.
Passages are short (71 terms on average after processing), so no further chunking is needed.
FiQA has **no titles** (all 57,638 are empty), so the title zone holds each passage's lead sentence
([ADR 0002](docs/adr/0002-lead-sentence-title-zone.md)).

## Which IR principle lives where

| principle | file |
|---|---|
| tokenisation, case folding, stop words, Porter stemming (each toggleable) | `src/text.py` |
| inverted index: dictionary, postings (doc, tf), df, doc lengths, title/body zones | `src/index.py` |
| positional postings and phrase queries; linear-merge postings intersection | `src/index.py` (`phrase`, `intersect`, `boolean_and`) |
| champion lists (top-r by tf) | `src/index.py` (`build_champions`), used in `src/sparse.py` |
| tf-idf VSM, SMART lnc.ltc, cosine, term-at-a-time accumulators, heap top-K | `src/sparse.py` (`TfidfRetriever`) |
| zone weighting `w_title·title + w_body·body` | `src/sparse.py` |
| Okapi BM25 (k1=1.2, b=0.75) | `src/sparse.py` (`BM25Retriever`) |
| dense retrieval, Reciprocal Rank Fusion | `src/dense.py` |
| P@k, Recall@k, nDCG@k, MRR@k | `src/metrics.py` |
| CiteGuard support = α·tf-idf cosine + (1−α)·term coverage; BM25 re-attribution | `src/citeguard.py` |
| experiments | `src/eval.py` |

Every retriever returns `(doc_id, score)` plus, with `explain=True`, a per-term breakdown whose sum is the score
(shown by `--show-scores`). The core IR is written from scratch: no Lucene, Pyserini, rank_bm25 or sklearn
vectorisers ([ADR 0001](docs/adr/0001-dependencies.md)).

## Results

All numbers: full corpus (57,638 docs), 200 seeded FiQA **test** queries, depth 100.
Anything tuned (zone weight, α, thresholds) was tuned on the **dev** split only.

### 1. Retrievers — `results/retrievers_full.{csv,png}`

| retriever | P@5 | P@10 | Recall@100 | nDCG@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|
| tf-idf lnc.ltc (baseline) | 0.072 | 0.051 | 0.513 | 0.173 | 0.194 | 3.9 |
| tf-idf + zones (w_title=0.1) | 0.072 | 0.049 | 0.509 | 0.173 | 0.207 | 4.7 |
| tf-idf + champion lists (r=200) | 0.056 | 0.040 | 0.411 | 0.154 | 0.168 | **0.33** |
| BM25 (k1=1.2, b=0.75) | 0.102 | 0.064 | 0.564 | 0.241 | 0.277 | 5.3 |
| dense (MiniLM-L6 + FAISS) | **0.154** | **0.095** | **0.725** | **0.376** | **0.446** | 9.3 |
| hybrid (RRF k=60, BM25 + dense) | 0.140 | 0.094 | 0.709 | 0.352 | 0.397 | 20.1 |

* **Sanity check:** our from-scratch BM25 scores nDCG@10 = 0.241, against the 0.236 reported for BM25 on FiQA in the BEIR paper.
* BM25's length normalisation and saturating tf beat lnc.ltc by +0.07 nDCG@10.
* Champion lists are **11.7× faster** than the full tf-idf scan but lose 0.10 Recall@100: the classic speed/quality trade-off.
* Zones: the dev sweep (`results/zones_dev_full.csv`) picked w_title = 0.1 (+0.016 nDCG@10 on dev). On test it is neutral (MRR +0.013, nDCG ±0), as expected for a "title" that duplicates the body (ADR 0002).
* Hybrid is *below* dense. Equal-weight RRF lets the much weaker BM25 run pull down the dense ranking. Weighted RRF is the obvious next step.

### 2. Ablation — `results/ablation_full.{csv,png}`

| | stop words removed + stemming | no stemming | no stop-word removal | neither |
|---|---|---|---|---|
| tf-idf nDCG@10 | **0.173** | 0.148 | 0.174 | 0.163 |
| BM25 nDCG@10 | **0.241** | 0.217 | 0.231 | 0.214 |
| BM25 ms/query | 5.3 | 3.8 | 31.7 | 30.4 |
| vocabulary | 54,837 | 75,409 | 54,927 | 75,557 |

Stemming cuts the vocabulary by 27% and adds ~0.02–0.025 nDCG@10. Removing stop words barely
changes quality, but makes queries **6× faster**, because stop words have the longest postings lists.

### 3. CiteGuard corruption test — `results/citeguard_*.{csv,png}`

Answers were generated (Groq `openai/gpt-oss-120b`, hybrid top-5) for 30 dev and 40 test queries.
Then 30% of citations were swapped to a different retrieved chunk at random (seeded), so we know
exactly which citations are wrong. A checker *flags* a sentence when support(sentence, cited chunk) < τ.
α and τ were tuned on dev (`citeguard_alpha_dev.csv`: best α = 0.75, τ = 0.18).

| checker (test: 152 sentences, 51 swapped) | precision | recall | F1 | flagged swaps whose original chunk was recovered |
|---|---|---|---|---|
| **lexical** (α·cosine + (1−α)·coverage) | **0.87** | 0.67 | **0.76** | **74%** (25/34) |
| NLI cross-encoder (deberta-v3-small) | 0.49 | 0.80 | 0.61 | 41% (17/41) |

Flagging everything gives F1 = 0.50. On test, the best lexical threshold would have been 0.22 (F1 0.84).
The dev-tuned 0.18 lands close to it (`citeguard_threshold_curve.png`). The NLI model's entailment
probabilities are near zero for most real answer sentences, because the LLM paraphrases and combines
long chunks, so its dev-tuned threshold collapses to 0.01. Our IR-based support score separates
right from wrong chunks better.

**Caveat.** This test measures *wrong-chunk* detection. With τ = 0.18, 68 of 70 sentences in real
(uncorrupted) answers are SUPPORTED, so subtle unsupported details inside an otherwise on-topic
sentence can pass. The manual-labelling step (`make labels` → `make agreement`) measures that case.

## What works / what's planned

Works: everything in the pipeline above, end to end in the CLI and the Streamlit page; all experiments
write CSV + PNG to `results/`; 31 unit tests, including lnc.ltc and BM25 scores on a toy corpus checked against
hand calculation (`tests/test_sparse.py`).

Planned / not done yet:
* Manual labels: `results/manual_labels.csv` has 70 sentences from 20 real answers, ready to label. Agreement (accuracy, Cohen's κ) is computed by `make agreement`.
* Weighted RRF, or tuning the RRF k on dev, so the hybrid at least matches dense.
* A sentence-level support score that also penalises unsupported numbers and entities (the main miss above).
* BM25F over the zones, instead of applying zone weighting to tf-idf only.

## AI use

This code was written with an AI coding agent (Claude Code, Claude Opus 5.5), working from my
specification and in stages: each stage was run, its output checked (e.g. full-corpus BM25 nDCG@10 against
the published BEIR figure), and errors fixed before moving on. Design decisions are recorded in `docs/adr/`.
The LLM used inside the system for answer generation is configurable (default: Groq, `openai/gpt-oss-120b`).
