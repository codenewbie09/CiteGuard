"""Module 2 — inverted index with title/body zones and positional body postings.

Layout (per zone): dictionary term → postings, where postings are two parallel arrays
(doc ids ascending, term frequencies). Body postings also carry positions, stored flat:
positions[term] = (offsets, flat) so posting i's positions are flat[offsets[i]:offsets[i+1]].
Parallel `array('I')` instead of lists of tuples keeps the full FiQA index small and fast to pickle.
"""
from __future__ import annotations

import math
import pickle
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from src.text import TextProcessor

ZONES = ("title", "body")


def intersect(p1: list[int], p2: list[int]) -> list[int]:
    """Linear merge of two ascending doc-id lists (IIR Fig. 1.6)."""
    i = j = 0
    out = []
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return out


def phrase_match(pos1, pos2) -> bool:
    """True if some position in pos2 directly follows one in pos1 (both ascending)."""
    i = j = 0
    while i < len(pos1) and j < len(pos2):
        if pos2[j] == pos1[i] + 1:
            return True
        if pos2[j] <= pos1[i]:
            j += 1
        else:
            i += 1
    return False


class InvertedIndex:
    def __init__(self, processor: TextProcessor):
        self.processor = processor
        self.doc_ids: list[str] = []
        self.docs: dict[str, dict[str, tuple[array, array]]] = {z: {} for z in ZONES}
        self.positions: dict[str, tuple[array, array]] = {}
        self.doc_len: dict[str, list[int]] = {z: [] for z in ZONES}
        self.lnc_norm: dict[str, list[float]] = {z: [] for z in ZONES}
        self.champion_r: int | None = None
        self._champions: dict[str, tuple[array, array]] = {}
        self.build_seconds = 0.0

    # ---------- construction ----------
    @classmethod
    def build(cls, docs, processor: TextProcessor, champion_r: int | None = None) -> InvertedIndex:
        """docs: iterable of {"id", "title", "body"}."""
        t0 = time.perf_counter()
        idx = cls(processor)
        raw = {z: defaultdict(lambda: (array("I"), array("I"))) for z in ZONES}
        pos_off: dict[str, array] = defaultdict(lambda: array("I", [0]))
        pos_flat: dict[str, array] = defaultdict(lambda: array("I"))

        for n, doc in enumerate(docs):
            idx.doc_ids.append(doc["id"])
            for zone in ZONES:
                terms = processor.process(doc[zone])
                tfs = Counter(terms)
                idx.doc_len[zone].append(len(terms))
                idx.lnc_norm[zone].append(math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in tfs.values())))
                for term, tf in tfs.items():
                    d, f = raw[zone][term]
                    d.append(n)
                    f.append(tf)
                if zone == "body":
                    where = defaultdict(list)
                    for p, term in enumerate(terms):
                        where[term].append(p)
                    for term, ps in where.items():
                        pos_flat[term].extend(ps)
                        pos_off[term].append(len(pos_flat[term]))

        idx.docs = {z: dict(raw[z]) for z in ZONES}
        idx.positions = {t: (pos_off[t], pos_flat[t]) for t in pos_flat}
        if champion_r:
            idx.build_champions(champion_r)
        idx.build_seconds = time.perf_counter() - t0
        return idx

    def build_champions(self, r: int) -> None:
        """Champion list per body term: the r postings with highest tf, kept in doc-id order."""
        self.champion_r = r
        self._champions = {t: self._top_r(t, r) for t in self.docs["body"]}

    def _top_r(self, term: str, r: int) -> tuple[array, array]:
        d, f = self.docs["body"][term]
        if len(d) <= r:
            return d, f
        keep = sorted(sorted(range(len(d)), key=lambda i: -f[i])[:r])
        return array("I", (d[i] for i in keep)), array("I", (f[i] for i in keep))

    # ---------- statistics ----------
    @property
    def N(self) -> int:
        return len(self.doc_ids)

    def avgdl(self, zone: str = "body") -> float:
        return sum(self.doc_len[zone]) / max(self.N, 1)

    def df(self, term: str, zone: str = "body") -> int:
        p = self.docs[zone].get(term)
        return len(p[0]) if p else 0

    def idf(self, term: str, zone: str = "body") -> float:
        """t in SMART 'ltc': log10(N / df)."""
        df = self.df(term, zone)
        return math.log10(self.N / df) if df else 0.0

    # ---------- access ----------
    def raw_postings(self, term: str, zone: str = "body") -> tuple[array, array]:
        return self.docs[zone].get(term, (array("I"), array("I")))

    def postings(self, term: str, zone: str = "body") -> list[tuple[int, int]]:
        d, f = self.raw_postings(term, zone)
        return list(zip(d, f))

    def champions(self, term: str, r: int | None = None) -> list[tuple[int, int]]:
        return list(zip(*self.raw_champions(term, r)))

    def raw_champions(self, term: str, r: int | None = None) -> tuple[array, array]:
        if r is None or r == self.champion_r:
            if term in self._champions:
                return self._champions[term]
        if term not in self.docs["body"]:
            return array("I"), array("I")
        return self._top_r(term, r or self.champion_r)

    def term_positions(self, term: str, i: int) -> array:
        off, flat = self.positions[term]
        return flat[off[i]:off[i + 1]]

    # ---------- boolean / phrase queries ----------
    def boolean_and(self, query: str) -> list[str]:
        terms = sorted(set(self.processor.process(query)), key=self.df)  # rarest first
        if not terms:
            return []
        result = list(self.raw_postings(terms[0])[0])
        for t in terms[1:]:
            result = intersect(result, list(self.raw_postings(t)[0]))
        return [self.doc_ids[d] for d in result]

    def phrase(self, query: str) -> list[str]:
        """Positional phrase query over the body zone (consecutive processed terms)."""
        terms = self.processor.process(query)
        if not terms or any(t not in self.positions for t in terms):
            return []
        # doc → index into each term's postings
        lookups = [{d: i for i, d in enumerate(self.raw_postings(t)[0])} for t in terms]
        candidates = set(lookups[0]).intersection(*lookups[1:])
        hits = []
        for doc in sorted(candidates):
            starts = set(self.term_positions(terms[0], lookups[0][doc]))
            for k in range(1, len(terms)):
                nxt = set(self.term_positions(terms[k], lookups[k][doc]))
                starts = {p for p in starts if p + k in nxt}
                if not starts:
                    break
            if starts:
                hits.append(self.doc_ids[doc])
        return hits

    # ---------- persistence and inspection ----------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        idx = cls.__new__(cls)
        with open(path, "rb") as f:
            idx.__dict__.update(pickle.load(f))
        return idx

    def stats(self) -> dict:
        s = {"documents": self.N, "build_seconds": round(self.build_seconds, 2)}
        for z in ZONES:
            s[f"{z}_vocab"] = len(self.docs[z])
            s[f"{z}_postings"] = sum(len(d) for d, _ in self.docs[z].values())
            s[f"{z}_avg_len"] = round(self.avgdl(z), 1)
        s["body_positions"] = sum(len(flat) for _, flat in self.positions.values())
        return s

    def describe_term(self, word: str, limit: int = 10, console=None) -> None:
        """Pretty-print df, idf and the head of each zone's postings list for a query word."""
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        console = console or Console()
        terms = self.processor.process(word) or [word]
        for term in terms:
            table = Table(title=f"term '{term}'  (from '{word}')", show_lines=False)
            table.add_column("zone")
            table.add_column("df", justify="right")
            table.add_column("idf=log10(N/df)", justify="right")
            table.add_column(escape(f"postings (doc_id:tf[positions]) — first {limit}"))
            for z in ZONES:
                d, f = self.raw_postings(term, z)
                items = []
                for i in range(min(limit, len(d))):
                    pos = ""
                    if z == "body":
                        ps = list(self.term_positions(term, i))
                        pos = f"{ps[:4]}{'…' if len(ps) > 4 else ''}"
                    items.append(f"{self.doc_ids[d[i]]}:{f[i]}{pos}")
                tail = " …" if len(d) > limit else ""
                table.add_row(z, str(len(d)), f"{self.idf(term, z):.3f}", escape(", ".join(items) + tail))
            console.print(table)
