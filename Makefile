PY := .venv/bin/python
FULL ?= --full          # `make eval FULL=` runs everything on the 5k-doc sample instead

.PHONY: setup data build-index test eval eval-retrievers eval-ablation eval-zones eval-citeguard labels agreement demo app

setup:                  ## create the venv and install dependencies
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

data:                   ## download BEIR FiQA into data/
	$(PY) -c "from src.config import load_config; from src.data import ensure_fiqa; print(ensure_fiqa(load_config()))"

build-index: data       ## build sample + full inverted indexes, print stats
	$(PY) -m src.build
	$(PY) -m src.build --full --term dividend "roth ira"

test:
	$(PY) -m pytest -q

eval: eval-retrievers eval-ablation eval-zones eval-citeguard

eval-retrievers:
	$(PY) -m src.eval retrievers $(FULL) --only tfidf tfidf+zones tfidf+champions bm25 dense hybrid
eval-ablation:
	$(PY) -m src.eval ablation $(FULL)
eval-zones:
	$(PY) -m src.eval zones $(FULL)
eval-citeguard:         ## uses cached answers in results/ if present (no API key needed)
	$(PY) -m src.eval citeguard $(FULL)

labels:
	$(PY) -m src.eval labels $(FULL)
agreement:
	$(PY) -m src.eval agreement

demo:
	$(PY) -m src.cli ask "Are Roth IRA withdrawals taxed?" --retriever hybrid --full --show-postings --show-scores

app:
	.venv/bin/streamlit run app.py
