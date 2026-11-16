"""
catalog.py — NASA Earthdata (CMR) metadata lookups that need no login.

  - collection search: which NASA collections (VIIRS, MODIS, ...) match a keyword, area and period;
  - granule search: the granules (e.g. VIIRS 6-minute swaths) of one collection intersecting a
    bbox in a time window, with their start times, i.e. the actual overpasses.

Metadata only: downloading granules (e.g. with earthaccess) needs an Earthdata login and is not
done here. Network access goes through an injectable async `get(url, params)` returning JSON.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import httpx

CMR_URL = "https://cmr.earthdata.nasa.gov/search"
HTTP_TIMEOUT_S = 60.0
MAX_GRANULES = 500

GetJson = Callable[[str, dict[str, Any] | None], Awaitable[Any]]


async def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _window(bbox: list[float] | None, start_date: str | None, end_date: str | None) -> dict[str, str]:
    params = {}
    if bbox:
        params["bounding_box"] = ",".join(str(x) for x in bbox)
    if start_date and end_date:
        params["temporal"] = f"{start_date[:10]}T00:00:00Z,{end_date[:10]}T23:59:59Z"
    return params


async def search_collections(keyword: str, bbox: list[float] | None = None,
                             start_date: str | None = None, end_date: str | None = None,
                             limit: int = 20, get: GetJson = http_get_json) -> dict[str, Any]:
    """NASA collections matching `keyword` (e.g. 'active fire', 'land surface temperature')
    whose extent overlaps bbox and the period. Returns {"total", "collections": [{short_name,
    version, title, provider, concept_id}]} or {"error": ...}."""
    params = {"keyword": keyword, "page_size": str(max(1, min(int(limit), 50))),
              **_window(bbox, start_date, end_date)}
    try:
        data = await get(f"{CMR_URL}/collections.umm_json", params)
    except Exception as exc:
        return {"error": f"Earthdata collection search failed: {exc}"}
    out = []
    for item in data.get("items", []):
        meta, umm = item.get("meta") or {}, item.get("umm") or {}
        out.append({"short_name": umm.get("ShortName"), "version": umm.get("Version"),
                    "title": umm.get("EntryTitle"), "provider": meta.get("provider-id"),
                    "concept_id": meta.get("concept-id"),
                    "abstract": " ".join(str(umm.get("Abstract") or "").split())[:600]})
    return {"total": data.get("hits", len(out)), "listed": len(out), "collections": out}


def _granule_start(umm: dict) -> str:
    temporal = umm.get("TemporalExtent") or {}
    return str((temporal.get("RangeDateTime") or {}).get("BeginningDateTime")
               or temporal.get("SingleDateTime") or "")


async def search_granules(short_name: str, bbox: list[float], start_date: str, end_date: str,
                          get: GetJson = http_get_json) -> dict[str, Any]:
    """Granules of collection `short_name` (e.g. 'VNP14IMG', 'MOD14') intersecting bbox
    [west, south, east, north] between start_date and end_date (YYYY-MM-DD, inclusive).
    Returns {short_name, start_date, end_date, total, listed, days: {day: [HH:MM, ...]}}
    (granule start times, UTC) or {"error": ...}. All versions of the short name are included."""
    start_date, end_date = start_date[:10], end_date[:10]
    params = {"short_name": short_name, "page_size": str(MAX_GRANULES),
              "sort_key": "start_date", **_window(bbox, start_date, end_date)}
    try:
        data = await get(f"{CMR_URL}/granules.umm_json", params)
    except Exception as exc:
        return {"error": f"Earthdata granule search failed: {exc}"}
    days: dict[str, list[str]] = {}
    items = data.get("items", [])
    for item in items:
        start = _granule_start(item.get("umm") or {})
        if len(start) >= 16:
            days.setdefault(start[:10], []).append(start[11:16])
    return {"short_name": short_name, "start_date": start_date, "end_date": end_date,
            "total": data.get("hits", len(items)), "listed": len(items),
            "days": {d: sorted(set(t)) for d, t in sorted(days.items())}}
