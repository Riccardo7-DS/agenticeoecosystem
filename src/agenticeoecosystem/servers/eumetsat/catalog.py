"""
catalog.py — EUMETSAT Data Store metadata lookups that need no login.

  - browse: every collection the Data Store holds (id, title, product count);
  - search-products: products of one collection intersecting a bbox in a time window.

Metadata only: downloading products needs an EUMETSAT account and is not done here.
Network access goes through an injectable async `get(url, params)` returning parsed JSON.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable

import httpx

EUMETSAT_API = "https://api.eumetsat.int/data"
BROWSE_URL = f"{EUMETSAT_API}/browse/1.0.0/collections"
SEARCH_URL = f"{EUMETSAT_API}/search-products/1.0.0/os"
HTTP_TIMEOUT_S = 60.0
MAX_PRODUCTS = 500          # one search page; enough for 3 days of 10-minute full-disk data

GetJson = Callable[[str, dict[str, Any] | None], Awaitable[Any]]


RETRIES = 3
MAX_CONCURRENT = 2                   # the Data Store rate-limits bursts (429)
_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _limiter() -> asyncio.Semaphore:
    """One semaphore per event loop (a semaphore must not be shared across loops)."""
    loop_id = id(asyncio.get_running_loop())
    return _SEMAPHORES.setdefault(loop_id, asyncio.Semaphore(MAX_CONCURRENT))


async def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with at most two requests in flight and a back-off retry on 429 Too Many Requests."""
    async with _limiter(), httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True) as client:
        for attempt in range(RETRIES + 1):
            resp = await client.get(url, params=params)
            if resp.status_code != 429 or attempt == RETRIES:
                resp.raise_for_status()
                return resp.json()
            try:
                wait = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                wait = 2.0 * (attempt + 1)
            await asyncio.sleep(min(wait, 30.0))


def collection_key(title: str) -> str:
    """Title without its format note, so 'Active Fire Monitoring (netCDF) - MTG' and
    'Active Fire Monitoring (CAP) - MTG' count as one collection."""
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", title or "")).strip().lower()


async def list_collections(query: str | None = None, get: GetJson = http_get_json) -> dict[str, Any]:
    """Collections in the EUMETSAT Data Store: {"total", "collections": [{id, title, products}]}.
    `query` keeps those whose title contains every word of it (case-insensitive)."""
    try:
        data = await get(BROWSE_URL, {"format": "json"})
    except Exception as exc:
        return {"error": f"EUMETSAT collection listing failed: {str(exc) or type(exc).__name__}"}
    words = (query or "").lower().split()
    out = []
    for link in data.get("links", []):
        title = link.get("datasetTitle") or ""
        if all(w in title.lower() for w in words):
            out.append({"id": link.get("title"), "title": title,
                        "products": link.get("numberOfProducts")})
    return {"total": len(data.get("links", [])), "matching": len(out), "collections": out}


async def search_products(collection_id: str, bbox: list[float], start_date: str, end_date: str,
                          get: GetJson = http_get_json) -> dict[str, Any]:
    """Products of `collection_id` (e.g. 'EO:EUM:DAT:0662') whose footprint intersects bbox
    [west, south, east, north] between start_date and end_date (YYYY-MM-DD, inclusive).
    Returns {collection, start_date, end_date, total, listed, days: {day: [HH:MM, ...]}}
    or {"error": ...}. Full-disk geostationary products always intersect a bbox on their
    disk: a count says the product exists, not that the area is usable at that viewing angle."""
    start_date, end_date = start_date[:10], end_date[:10]
    params = {"format": "json", "pi": collection_id, "bbox": ",".join(str(x) for x in bbox),
              "dtstart": f"{start_date}T00:00:00Z", "dtend": f"{end_date}T23:59:59Z",
              "c": str(MAX_PRODUCTS)}
    try:
        data = await get(SEARCH_URL, params)
    except Exception as exc:
        return {"error": f"EUMETSAT product search failed: {str(exc) or type(exc).__name__}"}
    days: dict[str, list[str]] = {}
    for feature in data.get("features", []):
        start = str((feature.get("properties") or {}).get("date") or "").split("/")[0]
        if len(start) >= 16:
            days.setdefault(start[:10], []).append(start[11:16])
    features = data.get("features", [])
    return {"collection": collection_id, "start_date": start_date, "end_date": end_date,
            "total": data.get("totalResults", len(features)), "listed": len(features),
            "days": {d: sorted(set(t)) for d, t in sorted(days.items())}}

