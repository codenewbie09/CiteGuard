#!/usr/bin/env bash
# Demo-video driver: runs each segment of the demo and waits for Enter in between.
#
#   ./demo.sh          run all segments
#   ./demo.sh 3        start at segment 3
#   ./demo.sh warm     load the index and models once (do this before recording)
#
# Needs: make setup, make build-index, and an LLM key in .env (segments 3 and 5 call the LLM).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1

# Q1 was probed twice with these exact settings: both runs flag the same sentence ("five-year rule",
# NLI ~0.003, missing anchor 5) as UNVERIFIED. Q2 is for the BM25 per-term breakdown; its verdicts vary.
Q1="Are Roth IRA withdrawals taxed?"
Q2="Can I deduct my home office if I am an employee?"

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)

title() { clear; printf '%s== %s ==%s\n\n' "$bold" "$1" "$reset"; }
shown() { local a out=(); for a in "$@"; do [[ $a == *" "* ]] && out+=("\"$a\"") || out+=("$a"); done; echo "${out[*]}"; }
run()   { printf '%s$ %s%s\n' "$bold" "$(shown "$@")" "$reset"; "$@" 2> >(grep -v -E "Loading weights|HF_TOKEN|Warning|warn" >&2); echo; }
pause() { read -r -p "${dim}[Enter: next segment]${reset} " _; }

if [[ "${1:-}" == "warm" ]]; then
  echo "warming caches (index, corpus, MiniLM, NLI model) …"
  $PY -c "from src.pipeline import RAG; RAG(full=True, retriever='hybrid', two_stage=True); print('ready')" 2>/dev/null
  exit 0
fi

if (( $(tput cols 2>/dev/null || echo 200) < 140 )); then
  echo "Terminal is $(tput cols) columns wide; widen it to 140+ (or shrink the font) so tables don't wrap."
  pause
fi

START=${1:-1}

seg1() {
  title "1. Inverted index built from scratch (57,638 FiQA passages)"
  run $PY -m src.build --full
}

seg2() {
  title "2. Postings lists: df, idf, tf and positions; title and body zones"
  run $PY -m src.cli postings "roth ira withdrawals" --full
}

seg3() {
  title "3. Ask: hybrid retrieval, cited answer, two-stage CiteGuard"
  run $PY -m src.cli ask "$Q1" --retriever hybrid --full --show-scores --two-stage
}

seg4() {
  title "4. Re-attribution: a deliberately wrong citation, repaired (cached answer, no LLM call)"
  run $PY -m src.cli reattribute
}

seg5() {
  title "5. Same pipeline with BM25: per-term score breakdown"
  run $PY -m src.cli ask "$Q2" --retriever bm25 --full --show-scores --two-stage
}

seg6() {
  title "6. Results (full corpus, FiQA test queries)"
  printf '%sRetrievers, all 648 test queries%s\n' "$bold" "$reset"
  column -s, -t results/retrievers_full_q648.csv; echo
  printf '%sWeighted hybrid vs dense: paired randomisation test%s\n' "$bold" "$reset"
  column -s, -t results/significance_full_q648.csv | head -5; echo
  printf '%sCiteGuard corruption test (30%% of citations swapped)%s\n' "$bold" "$reset"
  cut -d, -f1-8 results/citeguard_corruption.csv | column -s, -t; echo
  echo "Opening the plots …"
  for f in retrievers_full_q648 ablation_full citeguard_threshold_curve two_stage_curve; do
    xdg-open "results/$f.png" >/dev/null 2>&1 || true
  done
}

seg7() {
  title "7. Optional: the same thing in the Streamlit page"
  echo "Starting http://localhost:8501 (Ctrl+C to stop)"
  .venv/bin/streamlit run app.py
}

for i in 1 2 3 4 5 6 7; do
  if (( i < START )); then continue; fi
  "seg$i"
  if (( i < 7 )); then pause; fi
done
