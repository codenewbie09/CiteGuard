# 0002 — FiQA has no titles: use the lead sentence as the title zone

Status: accepted (2026-10-07)

**Context.** The spec asks for separate title and body zones. All 57,638 FiQA passages have an
empty `title` field (checked on load).

**Options.** (a) Drop zones. (b) Use the first sentence (≤30 words) of each passage as a
pseudo-title zone. (c) Generate titles with an LLM (cost, non-deterministic, not reproducible).

**Decision.** (b). It keeps zone weighting implementable and testable, is deterministic,
and the experiment answers a real question: does up-weighting the lead sentence help?

**Consequence.** The title zone duplicates text that is also in the body, so zone weighting
is expected to help less than with real titles. Measured result: see results/retrievers_*.csv.
