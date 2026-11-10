"""
catalog.py — Copernicus Climate Data Store (CDS) catalogue lookups that need no login.

Reads a dataset's catalogue entry (STAC collection): title, temporal coverage (ERA5 runs a few
days behind real time), update frequency and current availability. Retrieving data needs a CDS
account and is not done here. Network access goes through an injectable async `get(url, params)`.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import httpx

CDS_CATALOGUE = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections"
HTTP_TIMEOUT_S = 60.0
# Reanalysis datasets the inventory checks by default.
CORE_DATASETS = ("reanalysis-era5-single-levels", "reanalysis-era5-land",
                 "reanalysis-era5-pressure-levels")

GetJson = Callable[[str, dict[str, Any] | None], Awaitable[Any]]


async def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def dataset_info(dataset_id: str, start_date: str | None = None, end_date: str | None = None,
                       get: GetJson = http_get_json) -> dict[str, Any]:
    """Catalogue entry of a CDS dataset (e.g. 'reanalysis-era5-single-levels'). With a period,
    `covers_period` says whether the dataset's temporal extent includes it.
    Returns {dataset, title, start, end, update_frequency, status, covers_period} or {"error": ...}."""
    try:
        data = await get(f"{CDS_CATALOGUE}/{dataset_id}", None)
    except Exception as exc:
        return {"error": f"CDS catalogue lookup of '{dataset_id}' failed: {exc}"}
    interval = (((data.get("extent") or {}).get("temporal") or {}).get("interval") or [[None, None]])[0]
    first, last = ((interval[0] or "")[:10] or None), ((interval[1] or "")[:10] or None)
    info = {"dataset": data.get("id", dataset_id), "title": data.get("title"),
            "start": first, "end": last, "update_frequency": data.get("cads:update_frequency"),
            "status": (data.get("cads:sanity_check") or {}).get("status"),
            "disabled_reason": data.get("cads:disabled_reason")}
    if start_date and end_date:
        info["covers_period"] = bool((not first or first <= start_date[:10])
                                     and (not last or end_date[:10] <= last))
    return info
