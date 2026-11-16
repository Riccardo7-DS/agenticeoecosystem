"""
discovery.py — Federated data discovery: which collections, from any provider, could answer
the problem, and do they hold data over the study area and period.

  1. needs   the problem's data_needs (what must be observed; the PI writes them), else its
             objectives;
  2. list    every provider's collections, concurrently. CDSE STAC, the EUMETSAT Data Store and
             the CDS catalogue are listed whole (cached; none has usable free-text search) and
             filtered by extent. NASA CMR (tens of thousands of collections) is searched
             server-side with each need; its keyword search ANDs words, so a need with no hits
             is retried as consecutive word pairs;
  3. rank    embed the needs and each collection's title, keywords and description through an
             OpenAI-compatible /v1/embeddings endpoint (scripts/serve_embeddings.py by default)
             and keep the best collections per need plus the best per provider; word overlap
             if the embedding service is unreachable;
  4. verify  list the items / granules / products of the shortlist in the bbox and period.

There is no domain vocabulary in this module: relevance comes from the problem's needs, the
catalogues' own metadata and the embedding model, so new kinds of problems need no code change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:   # inventory imports this module: annotation-only import avoids a cycle
    from agenticeoecosystem.agents.virtual_lab.inventory import Catalogs
    from agenticeoecosystem.agents.virtual_lab.schemas import ScientificProblem

logger = logging.getLogger(__name__)

PROVIDER_LABELS = {
    "cdse": "Copernicus Data Space (STAC)",
    "eumetsat": "EUMETSAT Data Store",
    "earthdata": "NASA Earthdata (CMR)",
    "cds": "Copernicus Climate Data Store",
}
LIST_TTL_S = 12 * 3600       # whole-catalogue listings change rarely
CMR_PER_QUERY = 25           # NASA collections per keyword query
CMR_BACKOFF_QUERIES = 4      # word-pair retries per need with no CMR hits
PER_NEED = 4                 # best collections kept per need
PER_PROVIDER = 2             # best collections kept per provider (so none is crowded out)
MAX_SHORTLIST = 16
TEXT_CHARS = 900             # collection text embedded: title, keywords, description
# Minimum similarity for the per-provider quota (a provider with nothing relevant adds nothing).
MIN_PROVIDER_SCORE = {"embedding": 0.55, "lexical": 0.25}
# A collection counts for a need only if it scores at least this share of the need's best match,
# so a need with few good matches is not padded with weak ones.
RELATIVE_FLOOR = 0.8
# Hybrid ranking: embedding similarity + this weight x word overlap, so exact catalogue terms
# a small embedding model blurs ('multispectral', 'backscatter') still count. 0 = embeddings only.
DEFAULT_HYBRID_WEIGHT = 0.2


def hybrid_weight() -> float:
    """Weight of the lexical bonus added to embedding scores (env DISCOVERY_HYBRID_WEIGHT),
    read per ranking so a long-running app or a benchmark sweep picks up changes."""
    return float(os.getenv("DISCOVERY_HYBRID_WEIGHT", str(DEFAULT_HYBRID_WEIGHT)))

DEFAULT_EMBEDDING_ENDPOINT = "http://127.0.0.1:8090/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# call(tool, args, factory) runs one lookup through the lab's ledger -> (call id, result).
Call = Callable[[str, dict, Callable[[], Awaitable[Any]]], Awaitable[tuple[str | None, Any]]]


async def direct_call(tool: str, args: dict, factory: Callable[[], Awaitable[Any]]) -> tuple[None, Any]:
    """A Call that just runs the lookup (no ledger): for the agents' search_catalogues tool."""
    return None, await factory()


# ---------------------------------------------------------------------------------- data

@dataclass(frozen=True)
class Candidate:
    provider: str
    id: str                 # collection id / NASA short name / CDS dataset id
    title: str
    text: str               # what is embedded
    start: str | None = None
    end: str | None = None
    version: str | None = None   # NASA collections


@dataclass
class Hit:
    candidate: Candidate
    score: float
    need: str


@dataclass
class Discovery:
    needs: list[str]
    method: str                                      # "embedding" | "lexical" | "none"
    hits: list[Hit] = field(default_factory=list)
    listings: list[str] = field(default_factory=list)   # one line per provider listing
    checks: list[tuple[Hit, str | None, Any]] = field(default_factory=list)  # (hit, call id, result)


def _text(title: str, keywords: list[str], description: str) -> str:
    return " ".join(f"{title}. {', '.join(keywords)}. {description}".split())[:TEXT_CHARS]


def _norm_title(title: str) -> str:
    """Title without bracketed notes, so format/version variants of one product count once."""
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)|\[[^]]*\]", "", title or "")).strip().lower()


def needs_of(problem: ScientificProblem) -> list[str]:
    needs = [n.strip() for n in (getattr(problem, "data_needs", None) or []) if n and n.strip()]
    return needs or [o.strip() for o in problem.objectives if o.strip()][:4] or [problem.question]


# ------------------------------------------------------------------------------- listing

_LIST_CACHE: dict[tuple, tuple[float, Any]] = {}


async def _cached(key: tuple, factory: Callable[[], Awaitable[Any]]) -> Any:
    hit = _LIST_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < LIST_TTL_S:
        return hit[1]
    result = await factory()
    if isinstance(result, dict) and not result.get("error"):
        _LIST_CACHE[key] = (time.monotonic(), result)
    return result


def _from_listing(provider: str, result: Any) -> list[Candidate]:
    if not isinstance(result, dict) or result.get("error"):
        return []
    out = []
    for c in result.get("collections", []):
        if provider == "earthdata":
            cid, title = c.get("short_name"), c.get("title") or c.get("short_name") or ""
            text = _text(title, [], c.get("abstract") or "")
        else:
            cid, title = c.get("id"), c.get("title") or c.get("id") or ""
            text = _text(title, c.get("keywords") or [], c.get("description") or "")
        if cid:
            out.append(Candidate(provider, cid, title, text, c.get("start"), c.get("end"),
                                 c.get("version")))
    return out


# English function words only (no domain terms): they make poor keyword queries.
_FUNCTION_WORDS = frozenset("""the and for with from into onto over under per via its their this that
these those than then such each any all both also only which what when where while are was were
been being has have had not but can may use used using based""".split())


def _word_pairs(need: str) -> list[str]:
    """Consecutive content-word pairs, spread over the phrase (CMR keyword search ANDs words,
    so a long phrase finds nothing while its pairs do)."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", need)
             if len(w) > 2 and w.lower() not in _FUNCTION_WORDS]
    pairs = [f"{a} {b}" for a, b in pairwise(words)]
    if len(pairs) <= CMR_BACKOFF_QUERIES:
        return pairs
    step = (len(pairs) - 1) / (CMR_BACKOFF_QUERIES - 1)
    return [pairs[round(i * step)] for i in range(CMR_BACKOFF_QUERIES)]


async def list_candidates(needs: list[str], catalogs: Catalogs, bbox: list[float] | None, start: str | None,
                          end: str | None, call: Call) -> tuple[list[Candidate], list[str]]:
    """All providers' candidate collections for the needs, and one summary line per listing."""
    tasks: list[tuple[str, Awaitable]] = []
    if (cdse := getattr(catalogs, "cdse", None)) is not None and hasattr(cdse, "list_stac_collections"):
        args = {"bbox": bbox, "start_date": start, "end_date": end}
        # Cache keys name the client too: two clients for one provider never share listings.
        tasks.append(("cdse", call("list_stac_collections", args, lambda a=args: _cached(
            ("cdse", id(cdse), str(a)), lambda: cdse.list_stac_collections(**a)))))
    if (eum := getattr(catalogs, "eumetsat", None)) is not None and hasattr(eum, "list_collections"):
        tasks.append(("eumetsat", call("list_eumetsat_collections", {"query": None}, lambda: _cached(
            ("eumetsat", id(eum)), lambda: eum.list_collections(None)))))
    if (cds := getattr(catalogs, "cds", None)) is not None and hasattr(cds, "list_datasets"):
        args = {"start_date": start, "end_date": end}
        tasks.append(("cds", call("list_cds_datasets", args, lambda a=args: _cached(
            ("cds", id(cds), str(a)), lambda: cds.list_datasets(**a)))))
    nasa = getattr(catalogs, "earthdata", None)

    async def nasa_search(query: str) -> tuple[str | None, Any]:
        args = {"keyword": query, "bbox": bbox, "start_date": start, "end_date": end,
                "limit": CMR_PER_QUERY}
        return await call("search_earthdata_collections", args,
                          lambda a=args: nasa.search_collections(**a))

    async def nasa_for(need: str) -> list[tuple[str | None, Any]]:
        whole = await nasa_search(need)
        if isinstance(whole[1], dict) and whole[1].get("collections"):
            return [whole]
        # No hits for the whole phrase: its word pairs, together (ranking sorts them out).
        return [whole, *await asyncio.gather(*(nasa_search(q) for q in _word_pairs(need)))]

    nasa_tasks = [nasa_for(n) for n in needs] if nasa is not None and hasattr(nasa, "search_collections") else []
    results = await asyncio.gather(*(t for _, t in tasks), *nasa_tasks)
    candidates: list[Candidate] = []
    lines: list[str] = []
    for (provider, _), (cid, result) in zip(tasks, results[:len(tasks)], strict=True):
        found = _from_listing(provider, result)
        candidates += found
        ref = f"[{cid}] " if cid else ""
        if isinstance(result, dict) and result.get("error"):
            lines.append(f"- {ref}{PROVIDER_LABELS[provider]}: listing FAILED "
                         f"({str(result['error'])[:100]}); its collections were not considered.")
        else:
            total = result.get("total") if isinstance(result, dict) else None
            lines.append(f"- {ref}{PROVIDER_LABELS[provider]}: {len(found)} collections considered"
                         + (f" (of {total}; the rest do not overlap the area/period)"
                            if total and total != len(found) else ""))
    nasa_found, nasa_ids, nasa_failed = 0, [], 0
    for per_need in results[len(tasks):]:
        for cid, result in per_need:
            nasa_ids += [cid] if cid else []
            if isinstance(result, dict) and result.get("error"):
                nasa_failed += 1
            found = _from_listing("earthdata", result)
            nasa_found += len(found)
            candidates += found
    if nasa_tasks:
        ref = f"[{', '.join(nasa_ids)}] " if nasa_ids else ""
        lines.append(f"- {ref}{PROVIDER_LABELS['earthdata']}: {nasa_found} collections from "
                     f"keyword searches with the data needs"
                     + (f" ({nasa_failed} searches FAILED)" if nasa_failed else ""))
    unique = list({(c.provider, c.id): c for c in candidates}.values())
    return unique, lines


# ------------------------------------------------------------------------------- ranking

class EmbeddingClient:
    """OpenAI-compatible /v1/embeddings client with an in-process cache of document vectors."""

    def __init__(self, endpoint: str | None = None, model: str | None = None,
                 query_prefix: str | None = None, timeout_s: float = 60.0) -> None:
        self.endpoint = (endpoint or os.getenv("EMBEDDING_ENDPOINT") or DEFAULT_EMBEDDING_ENDPOINT).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        if query_prefix is None:
            query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX",
                                     BGE_QUERY_PREFIX if "bge" in self.model.lower() else "")
        self.query_prefix = query_prefix
        self.api_key = os.getenv("EMBEDDING_API_KEY", "")
        self.timeout_s = timeout_s
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        inputs = [f"{self.query_prefix}{t}" if query else t for t in texts]
        missing = [t for t in dict.fromkeys(inputs) if t not in self._cache]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for i in range(0, len(missing), 256):
                chunk = missing[i:i + 256]
                resp = await client.post(f"{self.endpoint}/embeddings", headers=headers,
                                         json={"model": self.model, "input": chunk})
                resp.raise_for_status()
                rows = sorted(resp.json()["data"], key=lambda r: r["index"])
                for text, row in zip(chunk, rows, strict=True):
                    self._cache[text] = _unit(row["embedding"])
        if len(self._cache) > 50_000:   # keep memory bounded in a long-running app
            self._cache = {t: self._cache[t] for t in inputs}
        return [self._cache[t] for t in inputs]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _lexical(needs: list[str], texts: list[str]) -> list[list[float]]:
    """Share of each need's words found in each text (fallback when embeddings are down)."""
    docs = [_tokens(t) for t in texts]
    rows = []
    for need in needs:
        words = _tokens(need)
        rows.append([len(words & d) / len(words) if words else 0.0 for d in docs])
    return rows


class EmbeddingsUnavailable(RuntimeError):
    """Raised by an embedder that cannot embed (switched off, or no service)."""


class NoEmbeddings:
    """Rank by word overlap only (offline tests; deterministic whatever services are up)."""

    async def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        raise EmbeddingsUnavailable("embeddings disabled")


NO_EMBEDDINGS = NoEmbeddings()
_DEFAULT_CLIENT: EmbeddingClient | None = None


def default_embedder() -> EmbeddingClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = EmbeddingClient()
    return _DEFAULT_CLIENT


async def rank(needs: list[str], candidates: list[Candidate],
               embedder: EmbeddingClient | None = None) -> tuple[list[Hit], str]:
    """Shortlist: the PER_NEED best per need, then the PER_PROVIDER best per provider above a
    minimum similarity, deduplicated by product title; best first, at most MAX_SHORTLIST."""
    if not needs or not candidates:
        return [], "none"
    texts = [c.text for c in candidates]
    try:
        embedder = embedder or default_embedder()
        queries = await embedder.embed(needs, query=True)
        docs = await embedder.embed(texts)
        scores = [[sum(a * b for a, b in zip(q, d)) for d in docs] for q in queries]
        if weight := hybrid_weight():
            lexical = _lexical(needs, texts)
            scores = [[s + weight * w for s, w in zip(row, lex)]
                      for row, lex in zip(scores, lexical)]
        method = "embedding"
    except (httpx.HTTPError, EmbeddingsUnavailable, KeyError, ValueError) as exc:
        if not isinstance(embedder, NoEmbeddings):
            logger.warning("Embedding service unavailable (%s): ranking by word overlap", exc)
        scores, method = _lexical(needs, texts), "lexical"

    best: dict[tuple[str, str], Hit] = {}

    def keep(i: int, j: int) -> None:
        c = candidates[i]
        key = (c.provider, _norm_title(c.title) or c.id)
        if key not in best or scores[j][i] > best[key].score:
            best[key] = Hit(c, round(scores[j][i], 3), needs[j])

    for j in range(len(needs)):
        order = sorted(range(len(candidates)), key=lambda i: -scores[j][i])
        floor = scores[j][order[0]] * RELATIVE_FLOOR
        for i in order[:PER_NEED * 3]:
            if (sum(h.need == needs[j] for h in best.values()) >= PER_NEED
                    or scores[j][i] <= 0 or scores[j][i] < floor):
                break
            keep(i, j)
    top = [max(range(len(needs)), key=lambda j: scores[j][i]) for i in range(len(candidates))]
    for provider in PROVIDER_LABELS:
        mine = sorted((i for i, c in enumerate(candidates) if c.provider == provider),
                      key=lambda i: -scores[top[i]][i])
        for i in mine[:PER_PROVIDER]:
            if scores[top[i]][i] >= MIN_PROVIDER_SCORE[method]:
                keep(i, top[i])
    hits = sorted(best.values(), key=lambda h: -h.score)[:MAX_SHORTLIST]
    return hits, method


# ---------------------------------------------------------------------------- verifying

def verification_call(hit: Hit, catalogs: Catalogs, bbox: list[float], start: str,
                      end: str) -> tuple[str, dict, Callable[[], Awaitable[Any]]] | None:
    """(tool, args, factory) listing the data of one shortlisted collection; None if the
    provider has no such lookup."""
    c = hit.candidate
    if c.provider == "cdse" and hasattr(catalogs.cdse, "search_stac_items"):
        args = {"collection": c.id, "bbox": bbox, "start_date": start, "end_date": end}
        return "search_stac_items", args, lambda: catalogs.cdse.search_stac_items(**args)
    if c.provider == "eumetsat" and hasattr(catalogs.eumetsat, "search_products"):
        args = {"collection_id": c.id, "bbox": bbox, "start_date": start, "end_date": end}
        return "count_eumetsat_products", args, lambda: catalogs.eumetsat.search_products(**args)
    if c.provider == "earthdata" and hasattr(catalogs.earthdata, "search_granules"):
        args = {"short_name": c.id, "bbox": bbox, "start_date": start, "end_date": end}
        return "search_earthdata_granules", args, lambda: catalogs.earthdata.search_granules(**args)
    if c.provider == "cds" and hasattr(catalogs.cds, "dataset_info"):
        args = {"dataset_id": c.id, "start_date": start, "end_date": end}
        return "cds_dataset_info", args, lambda: catalogs.cds.dataset_info(**args)
    return None


async def verify(hits: list[Hit], catalogs: Catalogs, bbox: list[float], start: str, end: str,
                 call: Call) -> list[tuple[Hit, str | None, Any]]:
    planned = [(h, verification_call(h, catalogs, bbox, start, end)) for h in hits]
    runs = [call(*spec) for _, spec in planned if spec]
    results = iter(await asyncio.gather(*runs))
    return [(h, *next(results)) if spec else (h, None, None) for h, spec in planned]


DataStatus = str   # "verified" | "partial" | "absent" | "unknown"


def data_status(result: Any) -> DataStatus:
    """Machine-readable verdict of a data lookup over the study area and period:
    verified (data found), partial (a global dataset not covering the whole period), absent
    (searched, none found) or unknown (failed, or no lookup). Missing metadata is unknown,
    never absent."""
    if not isinstance(result, dict) or result.get("error"):
        return "unknown"
    if "covers_period" in result:
        covers = result.get("covers_period")
        return "verified" if covers else "partial" if covers is not None else "unknown"
    if not any(k in result for k in ("total", "listed", "days", "products")):
        return "unknown"
    found = result.get("total") or result.get("listed") or result.get("days") or result.get("products")
    return "verified" if found else "absent"


def availability(result: Any) -> str:
    """One-phrase summary of a verification result (days lists, or a CDS catalogue entry)."""
    if result is None:
        return "not verified (no lookup for this provider)"
    if not isinstance(result, dict) or result.get("error"):
        error = str(result.get("error") if isinstance(result, dict) else result)[:100]
        return f"verification FAILED ({error}): availability unknown, NOT absent"
    if "covers_period" in result or "update_frequency" in result:
        covers = result.get("covers_period")
        verdict = ("covers the study period" if covers else "does NOT cover the whole study period"
                   if covers is not None else "coverage vs period unknown")
        return (f"{result.get('start')} to {result.get('end')}, updated "
                f"{result.get('update_frequency') or '?'}; {verdict} (global gridded dataset)")
    days = sorted((result.get("days") or {}).keys())
    listed, total = result.get("listed") or 0, result.get("total")
    if not listed and not total:
        return "NONE found in the bbox and period"
    count = f"{total}" if total else (f">={listed}" if listed else "?")
    if not days:
        return f"{count} items in the bbox and period"
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in pairwise(days)]
    revisit = f", median revisit {median(gaps):.0f} d" if gaps else ""
    partial = "" if not total or listed >= total else f" (first {listed} listed)"
    if total is None and listed:
        partial = " (newest listed only)"
    return f"{count} items on {len(days)} days, {days[0]}..{days[-1]}{revisit}{partial}"


# ------------------------------------------------------------------------------- driver

async def discover(problem: ScientificProblem, catalogs: Catalogs, call: Call,
                   embedder: EmbeddingClient | None = None) -> Discovery:
    """Run steps 1-4 for a ScientificProblem; verification needs a bbox and a valid period."""
    bbox = list(problem.study_area.bbox) if problem.study_area and problem.study_area.bbox else None
    period = problem.time_period
    start = end = None
    if period:
        try:
            start, end = date.fromisoformat(period.start[:10]), date.fromisoformat(period.end[:10])
            start, end = (start.isoformat(), end.isoformat()) if start <= end else (None, None)
        except ValueError:
            start = end = None
    return await discover_for(needs_of(problem), catalogs, bbox, start, end, call, embedder)


async def discover_for(needs: list[str], catalogs: Catalogs, bbox: list[float] | None, start: str | None,
                       end: str | None, call: Call, embedder: EmbeddingClient | None = None,
                       providers: list[str] | None = None, verify_data: bool = True) -> Discovery:
    if providers:
        catalogs = type(catalogs)(**{p: (getattr(catalogs, p) if p in providers else None)
                                     for p in PROVIDER_LABELS})
    candidates, listings = await list_candidates(needs, catalogs, bbox, start, end, call)
    hits, method = await rank(needs, candidates, embedder)
    found = Discovery(needs=needs, method=method, hits=hits, listings=listings)
    if hits and bbox and start and end and verify_data:
        found.checks = await verify(hits, catalogs, bbox, start, end, call)
    else:
        found.checks = [(h, None, None) for h in hits]
    return found


def render(found: Discovery, annotate: Callable[[Hit], str] | None = None) -> str:
    """The DATA DISCOVERY section of the inventory."""
    how = {"embedding": "semantic similarity (embedding model)",
           "lexical": "word overlap (embedding service unreachable: weaker ranking)",
           "none": "nothing to rank"}[found.method]
    lines = ["DATA DISCOVERY: collections from every catalogue ranked by " + how + " to the "
             "problem's data needs, then checked for data in the bbox and period:",
             "Data needs: " + "; ".join(f"({i}) {n}" for i, n in enumerate(found.needs, 1)),
             *found.listings]
    if not found.checks:
        lines.append("- No candidate collection was found; availability is unknown, NOT absent.")
    for hit, cid, result in found.checks:
        c = hit.candidate
        need_no = found.needs.index(hit.need) + 1 if hit.need in found.needs else "?"
        ref = f"[{cid}] " if cid else ""
        note = annotate(hit) if annotate else ""
        lines.append(f"- {ref}{PROVIDER_LABELS[c.provider]} · {c.id} — {c.title} "
                     f"(match {hit.score:.2f} to need {need_no}): {availability(result)}"
                     + (f"\n    {note}" if note else ""))
    lines.append("A collection missing here may still exist under another name: search_catalogues "
                 "searches all catalogues for any phrase.")
    return "\n".join(lines)


def as_tool_result(found: Discovery) -> dict[str, Any]:
    """search_catalogues' reply to an agent: the shortlist with availability."""
    return {"ranking": found.method, "listings": found.listings,
            "collections": [{"provider": PROVIDER_LABELS[h.candidate.provider], "id": h.candidate.id,
                             "title": h.candidate.title, "match": h.score,
                             "status": data_status(r), "availability": availability(r)}
                            for h, _, r in found.checks]}
