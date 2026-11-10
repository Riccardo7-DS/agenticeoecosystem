"""
catalog.py — Copernicus Data Space metadata lookups that need no login and no credits.

Two public endpoints:
  - the CDSE OData catalogue: which scenes (products) exist over a bbox on which dates;
  - the openEO collection list: which collections the CDSE openEO back-end offers, with
    their bands and temporal/spatial extent. Listing collections is free; loading or
    processing them (load_collection) needs an account and credits and is not used here.

Shared by the cdse_geocode MCP server and the Virtual Lab's data inventory, so both read the
catalogue the same way. Network access goes through an injectable async `get(url, params)`
returning parsed JSON, which keeps the functions testable offline.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

import httpx

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
OPENEO_URL = "https://openeo.dataspace.copernicus.eu/openeo/1.2"
HTTP_TIMEOUT_S = 60.0

CATALOG_ATTRIBUTES = ("productType", "cloudCover", "orbitDirection", "relativeOrbitNumber",
                      "platformSerialIdentifier")
# Sensors the CDSE catalogue does not hold -> the sub-agent that can list them.
ELSEWHERE = {"MTG": "eumetsat_agent", "FCI": "eumetsat_agent", "MSG": "eumetsat_agent",
             "SEVIRI": "eumetsat_agent", "VIIRS": "gee_agent", "MODIS": "gee_agent"}
PRODUCT_ALIASES = {"S2MSI2A": "MSIL2A", "S2MSI1C": "MSIL1C", "L2A": "MSIL2A", "L1C": "MSIL1C"}
# Instrument named in the collection -> product-name prefix on Sentinel-3.
_S3_INSTRUMENT_PREFIX = {"SLSTR": "SL_", "OLCI": "OL_", "SYN": "SY_"}
# Product types written into short collection names ('S1_GRD', 'SENTINEL1_SLC').
_PRODUCT_IN_NAME = ("GRD", "SLC", "OCN", "MSIL2A", "MSIL1C")

GetJson = Callable[[str, dict[str, Any] | None], Awaitable[Any]]


async def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GeocodeMCPServer"


async def _nominatim_get(url: str, params: dict[str, Any] | None = None) -> Any:
    """Nominatim requires an identifying User-Agent."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True,
                                 headers={"User-Agent": NOMINATIM_USER_AGENT}) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def geocode(place_name: str, limit: int = 3, get: GetJson = _nominatim_get) -> dict[str, Any]:
    """Bounding boxes [west, south, east, north] of the places matching `place_name`
    (OpenStreetMap Nominatim). Returns {place_name, results: [{display_name, bbox}]} or {"error"}."""
    try:
        found = await get(NOMINATIM_SEARCH_URL, {"q": place_name, "format": "json",
                                                 "limit": str(max(1, min(int(limit), 10)))})
    except Exception as exc:
        return {"error": f"Geocoding failed: {str(exc) or type(exc).__name__}"}
    results = []
    for r in found or []:
        south, north, west, east = (float(x) for x in r.get("boundingbox", [0, 0, 0, 0]))
        results.append({"display_name": r.get("display_name", ""),
                        "bbox": [round(west, 4), round(south, 4), round(east, 4), round(north, 4)]})
    if not results:
        return {"error": f"No results found for '{place_name}'."}
    return {"place_name": place_name, "results": results}


def normalise_collection(collection: str) -> tuple[str | None, str | None]:
    """Map a free-form name ('Sentinel-3 SLSTR', 'S1_GRD', 'SENTINEL2_L2A') to
    (catalogue collection, product-name filter implied by the name).
    Returns (None, reason) for a sensor the CDSE catalogue does not hold."""
    name = re.sub(r"[\s_]+", "-", collection.strip().upper())
    for sensor, agent in ELSEWHERE.items():
        if sensor in name:
            return None, (f"{sensor} is not in the CDSE catalogue. Query it through {agent}; "
                          "this does not mean the data do not exist.")
    match = re.search(r"(?:SENTINEL|S)-?(1|2|3|5P)(?![0-9])|LANDSAT-?(8|9)", name)
    if not match:
        return None, (f"Unknown collection '{collection}'. Use SENTINEL-1, SENTINEL-2, SENTINEL-3, "
                      "SENTINEL-5P, LANDSAT-8 or LANDSAT-9.")
    if match.group(2):
        return f"LANDSAT-{match.group(2)}", None
    number = match.group(1)
    if number == "3":
        return "SENTINEL-3", next((p for k, p in _S3_INSTRUMENT_PREFIX.items() if k in name), None)
    product = next((p for p in _PRODUCT_IN_NAME if p in name), None)
    if product is None and number == "2":
        product = PRODUCT_ALIASES.get(next((a for a in ("L2A", "L1C") if a in name), ""), None)
    return f"SENTINEL-{number}", product


def odata_filter(collection: str, bbox: list[float], start_date: str, end_date: str,
                 product_type: str | None, max_cloud_cover: float | None) -> str:
    west, south, east, north = bbox
    polygon = (f"POLYGON(({west} {south},{east} {south},{east} {north},"
               f"{west} {north},{west} {south}))")
    parts = [
        f"Collection/Name eq '{collection}'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        f"ContentDate/Start ge {start_date}T00:00:00.000Z",
        f"ContentDate/Start le {end_date}T23:59:59.999Z",
    ]
    if product_type:
        # The catalogue rejects contains() on attributes, but product names embed the type.
        escaped = product_type.replace("'", "''")
        parts.append(f"contains(Name,'{escaped}')")
    if max_cloud_cover is not None:
        parts.append("Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
                     f"and att/OData.CSC.DoubleAttribute/Value le {float(max_cloud_cover)})")
    return " and ".join(parts)


async def search_scenes(collection: str, bbox: list[float], start_date: str, end_date: str,
                        product_type: str | None = None, max_cloud_cover: float | None = None,
                        limit: int = 20, get: GetJson = http_get_json) -> dict[str, Any]:
    """Scenes in the CDSE catalogue intersecting bbox in [start_date, end_date].
    Returns {collection, product_type, bbox, start_date, end_date, total, listed, products}
    or {"error": ...}. `limit` is capped at 1000 (the catalogue's page size)."""
    resolved, implied = normalise_collection(collection)
    if resolved is None:
        return {"error": implied}
    product_type = PRODUCT_ALIASES.get((product_type or "").upper(), product_type) or implied
    start_date, end_date = start_date[:10], end_date[:10]
    params = {
        "$filter": odata_filter(resolved, bbox, start_date, end_date, product_type, max_cloud_cover),
        "$top": str(max(1, min(int(limit), 1000))),
        "$orderby": "ContentDate/Start asc",
        "$count": "true",
        "$expand": "Attributes",
    }
    try:
        data = await get(CDSE_ODATA_URL, params)
    except Exception as exc:
        return {"error": f"CDSE catalogue search failed: {exc}"}
    products = []
    for p in data.get("value", []):
        attrs = {a["Name"]: a["Value"] for a in p.get("Attributes", [])
                 if a.get("Name") in CATALOG_ATTRIBUTES}
        products.append({"name": p.get("Name"), "start": (p.get("ContentDate") or {}).get("Start"),
                         "online": p.get("Online"), **attrs})
    total = data.get("@odata.count", len(products))
    result = {"collection": resolved, "product_type": product_type, "bbox": bbox,
              "start_date": start_date, "end_date": end_date,
              "total": total, "listed": len(products), "products": products}
    if not total and product_type:
        result["note"] = (f"No product name contains '{product_type}'. Retry without product_type "
                          "before concluding that no data exist.")
    return result


def _overlaps_time(interval: list, start: str | None, end: str | None) -> bool:
    lo, hi = (interval + [None, None])[:2]
    return not ((end and lo and lo[:10] > end) or (start and hi and hi[:10] < start))


def _overlaps_space(extent: list, bbox: list[float] | None) -> bool:
    if not bbox or not extent or not any(extent):  # [0, 0, 0, 0] means unknown
        return True
    west, south, east, north = extent
    return not (bbox[2] < west or bbox[0] > east or bbox[3] < south or bbox[1] > north)


async def list_openeo_collections(bbox: list[float] | None = None, start_date: str | None = None,
                                  end_date: str | None = None,
                                  get: GetJson = http_get_json) -> dict[str, Any]:
    """Collections the CDSE openEO back-end offers, optionally only those whose declared
    extent overlaps bbox and [start_date, end_date]. Returns {"total", "matching",
    "collections": [{id, title, start, end}]} or {"error": ...}. Free: listing needs no login."""
    try:
        data = await get(f"{OPENEO_URL}/collections", None)
    except Exception as exc:
        return {"error": f"openEO collection listing failed: {exc}"}
    out = []
    for c in data.get("collections", []):
        extent = c.get("extent") or {}
        interval = ((extent.get("temporal") or {}).get("interval") or [[None, None]])[0] or [None, None]
        boxes = (extent.get("spatial") or {}).get("bbox") or [[]]
        if not (_overlaps_time(list(interval), start_date and start_date[:10], end_date and end_date[:10])
                and _overlaps_space(boxes[0], bbox)):
            continue
        out.append({"id": c.get("id"), "title": c.get("title"),
                    "keywords": [k for k in (c.get("keywords") or []) if isinstance(k, str)][:8],
                    "start": (interval[0] or "")[:10] or None, "end": (interval[1] or "")[:10] or None})
    return {"total": len(data.get("collections", [])), "matching": len(out), "collections": out}
