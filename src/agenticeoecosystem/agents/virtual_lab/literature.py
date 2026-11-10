"""
literature.py — Literature lookups for the Virtual Lab's literature check.

Two read-only operations the coordinator calls directly (not through an LLM),
so a contested claim is always checked against papers when the check runs:

  ArxivSearch               search_papers on the existing arxiv MCP server
                            (the toolset of arxiv_rag_agent), papers with abstracts.
  ScholarSearch             search_papers on scholar-search-mcp (Semantic Scholar + arXiv,
                            https://github.com/silung/scholar-search-mcp), an alternative
                            backend (VIRTUAL_LAB_LITERATURE_BACKEND=scholar). Results carry
                            citation counts, so its `rank` needs no extra lookup.
  rank_papers_by_citations  citation counts for arXiv ids, most cited first, from
                            Semantic Scholar, falling back to OpenAlex when its shared
                            unauthenticated quota is exhausted (OpenAlex can undercount
                            preprints whose citations go to the journal version).
                            arXiv itself has no citation data.

rank_papers_by_citations is also a plain function, so it can be handed to any
ADK agent as a FunctionTool. Ranking is applied to the top search results
(already relevance-ordered by arXiv), so it prefers well-cited papers among
relevant ones rather than the most-cited papers overall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_URL = "https://api.openalex.org/works"
_S2_FIELDS = "title,year,citationCount,influentialCitationCount"
# Claims at or above this word-overlap (Jaccard) are rewordings of one another. Calibrated on
# the 2026-11-01 EMSR859 run: three rewordings of one MODIS claim scored 0.43-0.65, while all
# distinct claims scored <= 0.37.
DUPLICATE_CLAIM_SIMILARITY = 0.4
_STOPWORDS = frozenset(["the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "is", "are", "be", "its", "it", "as", "by", "with", "but", "that", "this", "from", "at", "due", "may", "can"])
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


def normalise_arxiv_id(raw: str) -> str | None:
    """'arXiv:2401.12345v2' / 'http://arxiv.org/abs/2401.12345' -> '2401.12345'."""
    m = _ARXIV_ID.search(raw or "")
    return m.group(1) if m else None


def arxiv_ids_in(text: str) -> set[str]:
    return {m.group(1) for m in _ARXIV_ID.finditer(text or "")}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOPWORDS}


def claim_similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words."""
    wa, wb = _content_words(a), _content_words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def is_duplicate_claim(claim: str, seen: list[str]) -> bool:
    return any(claim_similarity(claim, s) >= DUPLICATE_CLAIM_SIMILARITY for s in seen)


_QUERY_SYNTAX = re.compile(r"\b(?:AND|OR|ANDNOT|NOT)\b|\b(?:ti|abs|au|cat|all):")


def query_terms(queries: list[str]) -> set[str]:
    """Content words of arXiv queries, without boolean operators and field prefixes."""
    return {w for q in queries for w in _content_words(_QUERY_SYNTAX.sub(" ", q)) if len(w) > 2}


def is_relevant(paper: dict, terms: set[str], min_terms: int = 2) -> bool:
    """True if title+abstract mention at least min_terms of the query terms (substring match,
    so 'fire' matches 'wildfires'). OR-queries match papers on a single generic word; this
    drops those."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return sum(t in text for t in terms) >= min(min_terms, len(terms))


def _split_top_level(query: str, op: str) -> list[str]:
    """Split on ` op ` outside parentheses and quotes."""
    parts, depth, quoted, start = [], 0, False, 0
    token = f" {op} "
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth = max(0, depth - 1)
        elif not quoted and depth == 0 and query.startswith(token, i):
            parts.append(query[start:i].strip())
            i += len(token)
            start = i
            continue
        i += 1
    parts.append(query[start:].strip())
    return [p for p in parts if p]


def limit_and(query: str) -> str:
    """At most one top-level AND: 'A AND B AND C' -> 'A AND (B OR C)'. Models chain ANDs
    despite being told not to, and in a category-restricted search that returns nothing."""
    clauses = _split_top_level(query, "AND")
    if len(clauses) <= 2:
        return query.strip()
    rest = " OR ".join(c[1:-1] if c.startswith("(") and c.endswith(")") else c for c in clauses[1:])
    return f"{clauses[0]} AND ({rest})"


def broaden_query(query: str) -> str | None:
    """Looser form of an arXiv query that returned nothing: drop phrase quotes and turn
    AND into OR. None if that changes nothing."""
    broad = re.sub(r"\bAND\b", "OR", query.replace('"', ""))
    broad = re.sub(r"\s+", " ", broad).strip()
    return broad if broad != query.strip() else None


async def rank_papers_by_citations(arxiv_ids: list[str]) -> dict[str, Any]:
    """Rank arXiv papers by citation count, most cited first.

    Args:
        arxiv_ids: arXiv identifiers exactly as returned by search_papers.

    Returns:
        {"ranked": [{"arxiv_id", "title", "year", "citation_count",
                     "influential_citation_count", "citations_per_year"}],
         "source": "semantic_scholar" | "openalex"} — papers without data keep
        their input order at the end with citation_count null; "error" is set
        if every source failed.
    """
    ids = list(dict.fromkeys(i for i in (normalise_arxiv_id(x) for x in arxiv_ids) if i))
    if not ids:
        return {"ranked": [], "source": "semantic_scholar"}
    async with httpx.AsyncClient(timeout=20) as client:
        payload, s2_error = await _semantic_scholar(client, ids)
        if payload is not None:
            return rank_from_payload(ids, payload, source="semantic_scholar")
        payload, oa_error = await _openalex(client, ids)
        if payload is not None:
            return rank_from_payload(ids, payload, source="openalex")
    return rank_from_payload(ids, None, f"{s2_error}; {oa_error}")


async def _semantic_scholar(client: httpx.AsyncClient, ids: list[str]) -> tuple[list | None, str | None]:
    headers = {"x-api-key": key} if (key := os.getenv("SEMANTIC_SCHOLAR_API_KEY")) else {}
    error = None
    for attempt in range(2):
        try:
            resp = await client.post(S2_BATCH_URL, params={"fields": _S2_FIELDS},
                                     json={"ids": [f"arXiv:{i}" for i in ids]}, headers=headers)
            if resp.status_code == 429:  # the unauthenticated pool is shared by everyone
                error = "rate limited by Semantic Scholar"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json(), None
        except (httpx.HTTPError, ValueError) as exc:
            return None, f"Semantic Scholar lookup failed: {exc}"
    return None, error


async def _openalex(client: httpx.AsyncClient, ids: list[str]) -> tuple[list | None, str | None]:
    """Same shape as the Semantic Scholar batch payload, via arXiv DOIs (10.48550/arXiv.<id>)."""
    try:
        resp = await client.get(OPENALEX_URL, params={
            "filter": "doi:" + "|".join(f"10.48550/arxiv.{i}" for i in ids),
            "select": "doi,title,publication_year,cited_by_count", "per-page": 50})
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"OpenAlex lookup failed: {exc}"
    by_id = {(w.get("doi") or "").lower().rsplit("arxiv.", 1)[-1]: w for w in works}
    return [({"title": w.get("title"), "year": w.get("publication_year"),
              "citationCount": w.get("cited_by_count")} if (w := by_id.get(i.lower())) else None)
            for i in ids], None


def rank_from_payload(ids: list[str], payload: list | None, error: str | None = None,
                      source: str = "semantic_scholar") -> dict[str, Any]:
    """Pure ranking step of rank_papers_by_citations (payload aligned with ids)."""
    this_year = date.today().year
    rows = []
    for i, arxiv_id in enumerate(ids):
        item = (payload[i] if payload and i < len(payload) else None) or {}
        count, year = item.get("citationCount"), item.get("year")
        rows.append({
            "arxiv_id": arxiv_id,
            "title": item.get("title"),
            "year": year,
            "citation_count": count,
            "influential_citation_count": item.get("influentialCitationCount"),
            "citations_per_year": (round(count / max(1, this_year - year + 1), 1)
                                   if count is not None and year else None),
            "_order": i,
        })
    rows.sort(key=lambda r: (r["citation_count"] is None, -(r["citation_count"] or 0), r["_order"]))
    for r in rows:
        del r["_order"]
    out: dict[str, Any] = {"ranked": rows, "source": source}
    if error:
        out["error"] = error
    return out


class ArxivSearch:
    """Calls search_papers on arxiv_rag_agent's MCP toolset, bypassing its LLM."""

    tool_name = "search_papers"

    def __init__(self, arxiv_agent, categories: list[str] | None = None) -> None:
        self._toolsets = [t for t in getattr(arxiv_agent, "tools", []) if hasattr(t, "get_tools")]
        self._tool = None
        self._categories = list(categories or [])  # server-side filter: any of these

    @classmethod
    def from_agents(cls, existing_agents: dict, categories: list[str] | None = None) -> ArxivSearch | None:
        agent = existing_agents.get("arxiv_rag_agent")
        search = cls(agent, categories) if agent is not None else None
        return search if search and search._toolsets else None

    async def _search_tool(self):
        if self._tool is None:
            for toolset in self._toolsets:
                for tool in await toolset.get_tools():
                    if tool.name == self.tool_name:
                        self._tool = tool
                        break
        if self._tool is None:
            raise RuntimeError("arxiv MCP server exposes no search_papers tool")
        return self._tool

    async def __call__(self, ctx, query: str, max_results: int) -> list[dict]:
        """Papers as dicts with id, title, abstract, published; raises on tool errors."""
        from google.adk.tools.tool_context import ToolContext

        tool = await self._search_tool()
        args = {"query": query, "max_results": max_results}
        if self._categories:
            args["categories"] = self._categories
        result = await tool.run_async(args=args, tool_context=ToolContext(ctx))
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        texts = [c.get("text", "") for c in (result or {}).get("content", []) if c.get("type") == "text"]
        if (result or {}).get("isError"):
            raise RuntimeError(" ".join(texts)[:500] or "search_papers failed")
        papers = []
        for text in texts:
            try:
                papers.extend(json.loads(text).get("papers", []))
            except (json.JSONDecodeError, AttributeError):
                continue
        for p in papers:
            p["id"] = normalise_arxiv_id(p.get("id", "")) or p.get("id", "")
            p["abstract"] = (p.get("abstract") or "").removeprefix("[EXTERNAL CONTENT] ")
        return papers


SCHOLAR_MCP_PACKAGE = "scholar-search-mcp==0.1.3"
# 0.1.3 requires only mcp>=0.9 but uses the decorator API that mcp 2.x removed
# ('Server' object has no attribute 'list_tools'), so the SDK is capped here.
SCHOLAR_MCP_SDK = "mcp<2"
# Variables scholar-search-mcp reads. The MCP SDK hands a stdio server only HOME/PATH/...,
# so these (the API key from .env in particular) must be forwarded explicitly.
_SCHOLAR_ENV = ("SEMANTIC_SCHOLAR_API_KEY", "SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR",
                "SCHOLAR_SEARCH_ENABLE_ARXIV", "SCHOLAR_SEARCH_CACHE_DIR",
                "SCHOLAR_SEARCH_CACHE_TTL_SECONDS", "UV_CACHE_DIR")
_S2_SEARCH_FIELDS = ["title", "abstract", "year", "publicationDate", "citationCount",
                     "influentialCitationCount", "externalIds"]


def plain_query(query: str) -> str:
    """Semantic Scholar's relevance search takes plain keywords, not arXiv query syntax:
    'ti:"burn severity" AND (dNBR OR RBR)' -> 'burn severity dNBR RBR'."""
    text = re.sub(r'[()"]', " ", _QUERY_SYNTAX.sub(" ", query))
    return re.sub(r"\s+", " ", text).strip()


def scholar_toolset():
    """McpToolset running scholar-search-mcp over stdio through uvx (pinned, in its own
    environment, so it adds no project dependency). The first start installs it (~1 min)."""
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp import StdioServerParameters
    from mcp.client.stdio import get_default_environment

    from agenticeoecosystem.utils.output_paths import server_output_dir

    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / "uvx")
    env = {**get_default_environment(), **{k: os.environ[k] for k in _SCHOLAR_ENV if os.environ.get(k)}}
    env.setdefault("SCHOLAR_SEARCH_CACHE_DIR", str(server_output_dir("scholar_search") / "cache"))
    return McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=uvx, args=["--from", SCHOLAR_MCP_PACKAGE, "--with", SCHOLAR_MCP_SDK,
                                       "python", "-m", "scholar_search_mcp"],
            env=env),
        timeout=300))


def scholar_paper(p: dict) -> dict | None:
    """A scholar-search-mcp paper in ArxivSearch's shape, or None without an arXiv id (the
    lab cites and links papers by arXiv id). Semantic Scholar hits keep it in externalIds;
    arXiv-channel hits use it as paperId."""
    raw = (p.get("externalIds") or {}).get("ArXiv") or (
        p.get("paperId", "") if p.get("source") == "arxiv" else "")
    arxiv_id = normalise_arxiv_id(raw) if raw else None
    if not arxiv_id:
        return None
    return {"id": arxiv_id, "title": p.get("title") or "", "abstract": p.get("abstract") or "",
            "published": p.get("publicationDate") or (str(p["year"]) if p.get("year") else ""),
            "year": p.get("year"), "citationCount": p.get("citationCount"),
            "influentialCitationCount": p.get("influentialCitationCount")}


class ScholarSearch:
    """Calls search_papers on scholar-search-mcp, bypassing any LLM. Same call shape as
    ArxivSearch; arXiv categories are not supported (Semantic Scholar has none)."""

    tool_name = "search_papers"

    def __init__(self, toolset=None) -> None:
        self._toolset = toolset if toolset is not None else scholar_toolset()
        self._tool = None
        self._seen: dict[str, dict] = {}   # arxiv id -> paper, for rank()

    async def _search_tool(self):
        if self._tool is None:
            self._tool = next((t for t in await self._toolset.get_tools() if t.name == self.tool_name), None)
        if self._tool is None:
            raise RuntimeError("scholar-search-mcp exposes no search_papers tool")
        return self._tool

    async def __call__(self, ctx, query: str, max_results: int) -> list[dict]:
        """Papers with an arXiv id, as dicts with id, title, abstract, published; raises on tool errors."""
        from google.adk.tools.tool_context import ToolContext

        tool = await self._search_tool()
        args = {"query": plain_query(query) or query, "limit": max_results, "fields": _S2_SEARCH_FIELDS}
        result = await tool.run_async(args=args, tool_context=ToolContext(ctx))
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        texts = [c.get("text", "") for c in (result or {}).get("content", []) if c.get("type") == "text"]
        if (result or {}).get("isError"):
            raise RuntimeError(" ".join(texts)[:500] or "search_papers failed")
        papers: list[dict] = []
        for text in texts:
            try:
                papers.extend(filter(None, map(scholar_paper, json.loads(text).get("data") or [])))
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        for p in papers:
            self._seen.setdefault(p["id"], p)
        return papers

    async def rank(self, arxiv_ids: list[str]) -> dict[str, Any]:
        """rank_papers_by_citations from the counts search_papers already returned; falls back
        to the Semantic Scholar/OpenAlex lookup if any paper came without one (arXiv-only hit)."""
        ids = list(dict.fromkeys(i for i in (normalise_arxiv_id(x) for x in arxiv_ids) if i))
        payload = [self._seen.get(i) for i in ids]
        if any(p is None or p.get("citationCount") is None for p in payload):
            return await rank_papers_by_citations(ids)
        return rank_from_payload(ids, payload, source="semantic_scholar")

    async def close(self) -> None:
        await self._toolset.close()
