"""Module 5 — generation: provider-agnostic LLM call, citation prompt, answer parser.

Provider and model come from .env (or the environment):
    LLM_PROVIDER=groq|anthropic|openai|gemini     LLM_MODEL=<optional override>
    GROQ_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible chat endpoints; Anthropic has its own Messages API shape.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "GEMINI_API_KEY", "gemini-2.5-flash"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
}

SYSTEM = (
    "You answer finance questions using ONLY the numbered evidence chunks provided. "
    "Write 3-6 sentences. End EVERY sentence with exactly one citation naming the single chunk "
    "that supports it, for example: 'Roth contributions are made with after-tax money [2].' "
    "Do not use outside knowledge. If the chunks do not contain enough information to answer, "
    "reply with exactly: Insufficient evidence."
)


def complete(prompt: str, system: str = SYSTEM, max_tokens: int = 600, temperature: float = 0.0) -> str:
    """One chat completion from the configured provider. Retries on rate limits."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    url, key_var, default_model = PROVIDERS[provider]
    key = os.getenv(key_var) or (provider == "gemini" and os.getenv("GOOGLE_API_KEY"))
    if not key:
        raise RuntimeError(f"set {key_var} in .env for LLM_PROVIDER={provider}")
    model = os.getenv("LLM_MODEL", default_model)
    if provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"model": model, "system": system, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
    else:
        headers = {"Authorization": f"Bearer {key}"}
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "citeguard/0.1", **headers})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
                continue
            raise RuntimeError(f"{provider} HTTP {e.code}: {e.read().decode()[:300]}") from e
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in data["content"])
    return data["choices"][0]["message"]["content"]


def build_prompt(question: str, chunks: list[str]) -> str:
    evidence = "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(chunks, 1))
    return f"Evidence chunks:\n\n{evidence}\n\nQuestion: {question}\n\nAnswer (every sentence ends with one citation):"


CITE_RE = re.compile(r"\s*\[(\d+)\]")
INSUFFICIENT_RE = re.compile(r"^\W*insufficient evidence\W*$", re.I)
SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


def _clean(s: str) -> str:
    s = s.strip().lstrip(".!?;,: ").strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def parse_answer(text: str) -> list[tuple[str, int | None]]:
    """Split an answer into (sentence, cited chunk number) pairs; numbers are 1-based.
    Text between two citations is one sentence; extra adjacent citations are dropped;
    trailing text without a citation is split into sentences with citation None."""
    text = " ".join(text.split())
    if not text or INSUFFICIENT_RE.match(text):
        return []
    out, pos = [], 0
    for m in CITE_RE.finditer(text):
        seg = _clean(text[pos:m.start()])
        pos = m.end()
        if seg.strip(".!? "):
            out.append((seg, int(m.group(1))))
    for seg in SENT_END_RE.split(text[pos:]):
        seg = _clean(seg)
        if seg.strip(".!? "):
            out.append((seg, None))
    return out
