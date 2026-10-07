# 0001 — Dependencies

Status: accepted (2026-10-07)

The core IR (tokeniser pipeline, inverted index, tf-idf, BM25, champion lists, metrics, CiteGuard
scoring) is written from scratch because the course grades classic IR principles.
Libraries are used only around it:

| need | library | why not hand-written |
|---|---|---|
| Porter stemmer | nltk | standard algorithm, not the graded part |
| embeddings | sentence-transformers (all-MiniLM-L6-v2) | dense baseline only |
| ANN index | faiss-cpu | inner-product search over 57k vectors |
| NLI checker | transformers cross-encoder | comparison baseline for CiteGuard |
| LLM | provider SDK via HTTP | generation is not graded |
| plots / tables | matplotlib, pandas, rich | presentation |

Rejected: rank_bm25, Pyserini/Lucene, sklearn TfidfVectorizer (would replace the graded part).
