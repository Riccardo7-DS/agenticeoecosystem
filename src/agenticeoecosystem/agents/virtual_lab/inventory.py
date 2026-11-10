"""
inventory.py — The lab's DATA INVENTORY: what data exist for the study area and period.

Built once, right after problem formulation, by plain function calls to public metadata APIs
(no LLM, no login, no processing credits), all run concurrently:
  - Copernicus Data Space: the openEO collection list (filtered to those relevant to the
    question) and catalogue scene searches for Sentinel-1/2/3 and Landsat, per acquisition day;
  - EUMETSAT Data Store: geostationary (MTG FCI, MSG SEVIRI) and Metop AVHRR product counts,
    plus the collections related to the question (e.g. MTG active fire, SLSTR FRP);
  - NASA Earthdata (CMR): granules, i.e. overpass times, of the VIIRS/MODIS products related to
    the question (e.g. active fire, burned area, land surface temperature);
  - Copernicus Climate Data Store: whether ERA5 / ERA5-Land cover the period.

Every participant sees the same inventory, so availability questions are answered once from
retrieved data instead of by each agent through sub-agent calls (slow, and the source of
false "no data" claims when a sub-agent's own searches were blocked).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from statistics import median
from typing import Any
from collections.abc import Awaitable, Callable

from agenticeoecosystem.servers.cdse_geocode.catalog import ELSEWHERE

from .sensor_catalog import (
    GEO_VZA_DEGRADED_DEG,
    GEO_VZA_UNUSABLE_DEG,
    SENSORS,
    geostationary_vza_deg,
    sensors_in_text,
)

SOURCE_PREFIX = "inventory"


@dataclass
class Catalogs:
    """Metadata clients the inventory and the direct tools use; None disables a provider.
    Each is a module (or test fake) with the functions of servers/<provider>/catalog.py."""
    cdse: Any = None
    eumetsat: Any = None
    earthdata: Any = None
    cds: Any = None

    @classmethod
    def public(cls) -> Catalogs:
        from agenticeoecosystem.servers.cdse_geocode import catalog as cdse
        from agenticeoecosystem.servers.climate import catalog as cds
        from agenticeoecosystem.servers.earthdata import catalog as earthdata
        from agenticeoecosystem.servers.eumetsat import catalog as eumetsat
        return cls(cdse=cdse, eumetsat=eumetsat, earthdata=earthdata, cds=cds)

    def __bool__(self) -> bool:
        return any((self.cdse, self.eumetsat, self.earthdata, self.cds))


@dataclass(frozen=True)
class SceneSearch:
    label: str
    collection: str
    product_type: str | None
    padded: bool        # widen the window by inventory_pad_days (pre-/post-event images)


SCENE_SEARCHES: tuple[SceneSearch, ...] = (
    SceneSearch("Sentinel-1 SAR (all product types)", "SENTINEL-1", None, True),
    SceneSearch("Sentinel-2 MSI L2A", "SENTINEL-2", "MSIL2A", True),
    SceneSearch("Sentinel-3 SLSTR L1 radiances", "SENTINEL-3", "SL_1_RBT", False),
    SceneSearch("Sentinel-3 OLCI L1 full resolution", "SENTINEL-3", "OL_1_EFR", False),
    SceneSearch("Landsat-8", "LANDSAT-8", None, True),
    SceneSearch("Landsat-9", "LANDSAT-9", None, True),
)
SCENE_LIMIT = 500   # scenes listed per search; the summary needs every one to be complete
# Product types that are not science data (auxiliary, raw) or duplicate another (cloud-optimised
# copies): left out of the per-day summary.
_HIDDEN_TYPES = ("_ETA_", "_RAW_", "-COG", "AUX_")
# openEO collections always shown: the missions the lab reasons about directly.
_CORE_COLLECTIONS = re.compile(r"^(SENTINEL|LANDSAT|COPERNICUS_30)")
# Words too generic to make a collection relevant to a question.
_GENERIC_STEMS = {"clms", "cope", "glob", "dail", "mont", "year", "vers", "land", "surf", "inde",
                  "with", "reso", "sate", "sent", "data", "prod", "euro", "mosa", "cond", "stat",
                  "term", "long", "shor", "regi", "nort", "hemi", "cont", "rast", "syst", "imag",
                  "proj", "seas", "base", "laye", "esa", "vito", "time", "seri", "cog", "nrt", "area"}
MAX_EXTRA_COLLECTIONS = 25


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    # Compounds carry their content word at the end: wildfire -> fire, firestorm -> fire.
    extra = {"fire" for w in words if "fire" in w}
    return ({w[:4] for w in words} | extra) - _GENERIC_STEMS


def wanted_stems(text: str) -> set[str]:
    """Content stems of the question, with related concepts added: a question about ignition
    or burned area is a fire question even if it never says 'fire'."""
    stems = _stems(text)
    for concept in _CONCEPTS:
        if stems & concept:
            stems |= concept
    return stems


def relevant_collections(collections: list[dict], problem_text: str) -> tuple[list[str], int]:
    """Collection ids worth showing for this problem: the core missions, plus collections whose
    title or keywords share a content word (4-letter stem: burned/burnt -> burn) with the
    problem. Returns (ids, number left out)."""
    wanted = wanted_stems(problem_text)
    core = [c["id"] for c in collections if _CORE_COLLECTIONS.match(c.get("id") or "")]
    extra = [c["id"] for c in collections if c["id"] not in core
             and _stems(" ".join([c.get("title") or "", *c.get("keywords", [])])) & wanted]
    extra = extra[:MAX_EXTRA_COLLECTIONS]
    return core + extra, len(collections) - len(core) - len(extra)


def valid_period(start: str | None, end: str | None) -> bool:
    try:
        return date.fromisoformat((start or "")[:10]) <= date.fromisoformat((end or "")[:10])
    except ValueError:
        return False


def search_window(problem_start: str, problem_end: str, search: SceneSearch,
                  pad_days: int) -> tuple[str, str]:
    """Padded searches cover pad_days either side; unpadded ones (frequent revisits such
    as Sentinel-3) one day either side, which keeps their scene lists complete."""
    pad = timedelta(days=pad_days if search.padded else 1)
    start, end = date.fromisoformat(problem_start[:10]), date.fromisoformat(problem_end[:10])
    return (start - pad).isoformat(), (end + pad).isoformat()


def summarise_scenes(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per acquisition day: scene count, product types, cloud-cover range, UTC start times."""
    days: dict[str, dict[str, Any]] = {}
    for p in result.get("products", []):
        start = p.get("start") or ""
        if len(start) < 10 or any(h in str(p.get("productType") or "") for h in _HIDDEN_TYPES):
            continue
        day = days.setdefault(start[:10], {"n": 0, "types": set(), "cloud": [], "times": set()})
        day["n"] += 1
        if p.get("productType"):
            day["types"].add(p["productType"])
        if isinstance(p.get("cloudCover"), (int, float)):
            day["cloud"].append(float(p["cloudCover"]))
        if len(start) >= 16:
            day["times"].add(start[11:16])
    return dict(sorted(days.items()))


def _day_text(day: str, info: dict[str, Any]) -> str:
    parts = [f"{day} ({info['n']})"]
    if info["types"]:
        parts.append(",".join(sorted(info["types"])))
    if info["cloud"]:
        lo, hi = min(info["cloud"]), max(info["cloud"])
        parts.append(f"cloud {lo:.0f}%" if round(lo) == round(hi) else f"cloud {lo:.0f}-{hi:.0f}%")
    times = sorted(info["times"])
    if times:
        parts.append(" ".join(times[:6]) + (" ..." if len(times) > 6 else "") + " UTC")
    return " ".join(parts)


def _event_groups(days: dict[str, Any], event: tuple[str, str] | None) -> list[tuple[str, list[str]]]:
    """Days split into BEFORE / DURING / AFTER the event period (one unlabelled group without it)."""
    if not event:
        return [("", list(days))]
    return [("BEFORE the event", [d for d in days if d < event[0]]),
            ("DURING the event", [d for d in days if event[0] <= d <= event[1]]),
            ("AFTER the event", [d for d in days if d > event[1]])]


def format_scene_row(call_id: str, search: SceneSearch, result: dict[str, Any] | None,
                     window: tuple[str, str], event: tuple[str, str] | None = None) -> str:
    """One catalogue search; with `event` (start, end) the days are split into
    before / during / after it, so pre- and post-event images are explicit."""
    head = f"- [{call_id}] {search.label}, {window[0]}..{window[1]}"
    if not result or result.get("error"):
        error = (result or {}).get("error", "no result")
        return f"{head}: SEARCH FAILED ({error[:120]}). Availability unknown, NOT absent."
    total, listed = result.get("total", 0), result.get("listed", 0)
    if not total:
        return f"{head}: no scenes in the catalogue."
    days = summarise_scenes(result)
    note = "" if listed == total else f" (first {listed} of {total} listed; later days not shown)"
    lines = []
    for label, group in _event_groups(days, event):
        body = " | ".join(_day_text(d, days[d]) for d in group) if group else "none"
        lines.append(f"{label}: {body}" if label else body)
    return f"{head}: {total} scenes on {len(days)} days{note}\n    " + "\n    ".join(lines)


def format_inventory(problem_area: str, bbox: list[float] | None, scene_rows: list[str],
                     openeo: dict[str, Any] | None, openeo_call_id: str | None,
                     problem_text: str = "", other_sections: list[str] = ()) -> str:
    lines = [f"Study area: {problem_area} bbox {bbox}" if bbox else
             "No study-area bbox, so no scene or product searches were run."]
    if scene_rows:
        lines.append("COPERNICUS DATA SPACE: catalogue scenes intersecting the bbox, per acquisition "
                     "day (count, product types, cloud cover, start times):")
        lines.extend(scene_rows)
    if openeo_call_id and openeo and not openeo.get("error"):
        ids, hidden = relevant_collections(openeo.get("collections", []), problem_text)
        more = (f" (+{hidden} more not related to the question; list_openeo_collections shows all)"
                if hidden else "")
        lines.append(f"- [{openeo_call_id}] openEO collections on the Copernicus Data Space whose "
                     f"extent overlaps the area and period, core missions and those related to "
                     f"the question (listing is free, loading them needs credits, so they are "
                     f"plan-only here): {', '.join(ids)}{more}")
    elif openeo_call_id:
        lines.append(f"- [{openeo_call_id}] openEO collection listing FAILED; collection "
                     "availability unknown.")
    lines.extend(other_sections)
    if other_sections:
        lines.append("NOT in this inventory (its silence about them says nothing): sensors and "
                     "providers not listed above, e.g. GOES ABI, Himawari AHI, commercial imagery.")
    else:
        elsewhere: dict[str, list[str]] = {}
        for sensor, agent in ELSEWHERE.items():
            elsewhere.setdefault(agent, []).append(sensor)
        lines.append("NOT in this inventory (its silence about them says nothing): "
                     + "; ".join(f"{', '.join(s)} -> ask {a}" for a, s in elsewhere.items()) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------- other providers

@dataclass(frozen=True)
class ProductSearch:
    key: str            # EUMETSAT collection id or NASA short name
    label: str
    stems: frozenset[str] = frozenset()   # empty: always searched; else only if the question shares one
    sub_satellite_lon: float | None = None  # geostationary products: where the satellite sits


_FIRE = frozenset({"fire", "burn", "igni", "wild", "hots", "smok"})
_HEAT = frozenset({"temp", "lst", "heat", "ther"})
_CONCEPTS = (_FIRE,)
# Words that name what a product observes. EUMETSAT titles are matched on these only: their
# other words (instrument, level, region, format) overlap with any question by chance.
_DOMAIN_STEMS = _FIRE | _HEAT | frozenset({
    "clou", "snow", "aero", "dust", "ozon", "prec", "rain", "wind", "soil", "mois", "vege",
    "ndvi", "flood", "wate", "ocea", "sea", "ice", "volc", "ash", "smoke", "trac", "nitr",
    "carb", "meth", "sulp", "humi", "radi", "albe", "evap", "lake", "chlo"})
EUMETSAT_CORE: tuple[ProductSearch, ...] = (
    ProductSearch("EO:EUM:DAT:0662", "MTG FCI L1c normal resolution (0 deg, 10 min)",
                  sub_satellite_lon=0.0),
    ProductSearch("EO:EUM:DAT:MSG:HRSEVIRI", "MSG SEVIRI L1.5 (0 deg, 15 min)",
                  sub_satellite_lon=0.0),
    ProductSearch("EO:EUM:DAT:METOP:AVHRRL1", "Metop AVHRR L1B"),
)
MAX_EUMETSAT_EXTRA = 3
# EUMETSAT collections that are long-term records, not the event's data.
_RECORD = re.compile(r"climate data record|data record|fundamental data record|reprocess", re.I)
NASA_PRODUCTS: tuple[ProductSearch, ...] = (
    ProductSearch("VNP14IMG", "VIIRS S-NPP active fire 375 m", _FIRE),
    ProductSearch("VJ114IMG", "VIIRS NOAA-20 active fire 375 m", _FIRE),
    ProductSearch("VJ214IMG", "VIIRS NOAA-21 active fire 375 m", _FIRE),
    ProductSearch("MOD14", "MODIS Terra thermal anomalies/fire 1 km", _FIRE),
    ProductSearch("MYD14", "MODIS Aqua thermal anomalies/fire 1 km", _FIRE),
    ProductSearch("MCD64A1", "MODIS burned area monthly 500 m", frozenset({"burn", "scar"})),
    ProductSearch("MOD11A1", "MODIS Terra land surface temperature daily 1 km", _HEAT - {"ther"}),
    ProductSearch("VNP21A1D", "VIIRS S-NPP land surface temperature daily 1 km", _HEAT - {"ther"}),
)
CDS_DATASETS = ("reanalysis-era5-single-levels", "reanalysis-era5-land")


def _wanted(search: ProductSearch, stems: set[str]) -> bool:
    return not search.stems or bool(search.stems & stems)


def eumetsat_extras(collections: list[dict], problem_text: str) -> list[ProductSearch]:
    """EUMETSAT collections related to the question beyond the core ones, one per product
    (format variants such as netCDF/GRIB merged), long-term data records left out."""
    from agenticeoecosystem.servers.eumetsat.catalog import collection_key
    wanted, core = wanted_stems(problem_text), {s.key for s in EUMETSAT_CORE}
    seen, out = set(), []
    for c in collections:
        title, key = c.get("title") or "", collection_key(c.get("title") or "")
        if (c.get("id") in core or key in seen or _RECORD.search(title)
                or not _stems(title) & wanted & _DOMAIN_STEMS):
            continue
        seen.add(key)
        out.append(ProductSearch(c["id"], title, sub_satellite_lon=geostationary_lon(title)))
        if len(out) == MAX_EUMETSAT_EXTRA:
            break
    return out


_IODC = re.compile(r"indian ocean|iodc|41\.5|45\.5", re.I)
IODC_LON = 45.5


def geostationary_lon(title: str) -> float | None:
    """Sub-satellite longitude of a EUMETSAT geostationary product, from its title
    (MTG/MSG at 0 deg unless the title names the Indian Ocean service); None if not geostationary."""
    sensors = [s for s in sensors_in_text(title) if s.orbit == "geostationary"]
    if not sensors:
        return None
    return IODC_LON if _IODC.search(title) else sensors[0].sub_satellite_lon


def _centre(bbox: list[float]) -> tuple[float, float]:
    return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2


def geometry_note(bbox: list[float] | None, sub_satellite_lon: float | None) -> str:
    """Viewing-angle verdict for a geostationary product over the bbox centre ('' if n/a)."""
    if not bbox or sub_satellite_lon is None:
        return ""
    vza = geostationary_vza_deg(*_centre(bbox), sub_satellite_lon)
    where = f"viewing zenith angle {vza:.0f} deg from the satellite at {sub_satellite_lon:+.1f} deg"
    if vza > GEO_VZA_UNUSABLE_DEG:
        return (f"UNUSABLE GEOMETRY: {where} (> {GEO_VZA_UNUSABLE_DEG:.0f}): the data exist but "
                "this area is at the disk edge; do not plan quantitative use.")
    if vza > GEO_VZA_DEGRADED_DEG:
        return (f"DEGRADED GEOMETRY: {where} (> {GEO_VZA_DEGRADED_DEG:.0f}): large pixels and long "
                "slant paths; usable with caution.")
    return f"Geometry OK: {where}."


def best_geostationary(bbox: list[float]) -> str:
    """The catalog's geostationary imager with the lowest viewing angle over the bbox."""
    lat, lon = _centre(bbox)
    geo = [(geostationary_vza_deg(lat, lon, s.sub_satellite_lon), s) for s in SENSORS
           if s.orbit == "geostationary" and s.sub_satellite_lon is not None]
    vza, s = min(geo, key=lambda x: x[0])
    access = ("fetch via " + ", ".join(s.sub_agents)) if s.sub_agents else "no sub-agent in this system fetches it"
    return f"Best-placed geostationary imager for this area: {s.key} (viewing zenith angle {vza:.0f} deg; {access})."


def _times_text(times: list[str]) -> str:
    if len(times) <= 8:
        return " ".join(times) + " UTC"
    minutes = [int(t[:2]) * 60 + int(t[3:5]) for t in times]
    step = median(b - a for a, b in pairwise(minutes))
    return f"{times[0]}-{times[-1]} UTC, every ~{step:.0f} min"


def format_days_row(call_id: str, label: str, result: dict[str, Any] | None,
                    window: tuple[str, str], unit: str = "products", geometry: str = "") -> str:
    """Row for a product/granule search returning {total, listed, days: {day: [HH:MM]}};
    `geometry` (see geometry_note) is appended when products were found."""
    head = f"- [{call_id}] {label}, {window[0]}..{window[1]}"
    if not result or result.get("error"):
        error = str((result or {}).get("error", "no result"))
        return f"{head}: SEARCH FAILED ({error[:120]}). Availability unknown, NOT absent."
    total, listed, days = result.get("total", 0), result.get("listed", 0), result.get("days") or {}
    if not total:
        return f"{head}: none found."
    note = "" if listed == total else f" (first {listed} of {total} listed)"
    return (f"{head}: {total} {unit}{note}\n    "
            + " | ".join(f"{d} ({len(t)}) {_times_text(t)}" for d, t in days.items())
            + (f"\n    {geometry}" if geometry else ""))


def format_cds_row(call_id: str, info: dict[str, Any] | None) -> str:
    if not info or info.get("error"):
        error = str((info or {}).get("error", "no result"))
        return f"- [{call_id}] CDS lookup FAILED ({error[:120]}). Availability unknown, NOT absent."
    covers = info.get("covers_period")
    verdict = "" if covers is None else ("; covers the study period" if covers
                                         else "; does NOT yet cover the whole study period")
    return (f"- [{call_id}] {info.get('title')} ({info.get('dataset')}): {info.get('start')} to "
            f"{info.get('end')}, updated {info.get('update_frequency') or '?'}, status "
            f"{info.get('status') or '?'}{verdict}")


# call(tool, args, factory) runs one lookup through the lab's ledger and returns
# (ledger call id, result); (None, None) when the call was not made.
Call = Callable[[str, dict, Callable[[], Awaitable[Any]]], Awaitable[tuple[str | None, Any]]]


async def build_inventory(problem, catalogs: Catalogs, call: Call, pad_days: int) -> str:
    """Run every provider's lookups concurrently (EUMETSAT searches after its collection list)
    and return the DATA INVENTORY text shown to all participants."""
    bbox = list(problem.study_area.bbox) if problem.study_area and problem.study_area.bbox else None
    period = problem.time_period
    start, end = ((period.start[:10], period.end[:10])
                  if period and valid_period(period.start, period.end) else (None, None))
    text = " ".join([problem.question, *problem.objectives])
    stems = wanted_stems(text)
    searchable = bool(bbox and start)

    async def none():
        return None, None

    # Stage 1 ---------------------------------------------------------------
    cdse, eum, nasa, cds = catalogs.cdse, catalogs.eumetsat, catalogs.earthdata, catalogs.cds
    openeo_args = {"bbox": bbox, "start_date": start, "end_date": end}
    openeo_task = (call("list_openeo_collections", openeo_args,
                        lambda: cdse.list_openeo_collections(**openeo_args)) if cdse else none())
    scenes, scene_windows = [], []
    if cdse and searchable:
        for search in SCENE_SEARCHES:
            window = search_window(start, end, search, pad_days)
            args = {"collection": search.collection, "bbox": bbox, "start_date": window[0],
                    "end_date": window[1], "product_type": search.product_type, "limit": SCENE_LIMIT}
            scenes.append(call("search_cdse_catalog", args, lambda a=args: cdse.search_scenes(**a)))
            scene_windows.append(window)
    eum_list_task = (call("list_eumetsat_collections", {"query": None},
                          lambda: eum.list_collections(None)) if eum else none())
    nasa_searches = [p for p in NASA_PRODUCTS if _wanted(p, stems)] if nasa and searchable else []
    nasa_tasks = []
    for p in nasa_searches:
        args = {"short_name": p.key, "bbox": bbox, "start_date": start, "end_date": end}
        nasa_tasks.append(call("search_earthdata_granules", args, lambda a=args: nasa.search_granules(**a)))
    cds_tasks = []
    for dataset in (CDS_DATASETS if cds else ()):
        args = {"dataset_id": dataset, "start_date": start, "end_date": end}
        cds_tasks.append(call("cds_dataset_info", args, lambda a=args: cds.dataset_info(**a)))
    stage1 = await asyncio.gather(openeo_task, eum_list_task, *scenes, *nasa_tasks, *cds_tasks)
    (oe_id, oe_result), (eum_list_id, eum_list) = stage1[0], stage1[1]
    rest = stage1[2:]
    scene_results, rest = rest[:len(scenes)], rest[len(scenes):]
    nasa_results, cds_results = rest[:len(nasa_tasks)], rest[len(nasa_tasks):]

    # Stage 2: EUMETSAT product searches (extras chosen from its collection list) ---------
    eum_searches: list[ProductSearch] = []
    if eum and searchable:
        listed = (eum_list or {}).get("collections", []) if isinstance(eum_list, dict) else []
        eum_searches = [*EUMETSAT_CORE, *eumetsat_extras(listed, text)]
    eum_tasks = []
    for p in eum_searches:
        args = {"collection_id": p.key, "bbox": bbox, "start_date": start, "end_date": end}
        eum_tasks.append(call("count_eumetsat_products", args, lambda a=args: eum.search_products(**a)))
    eum_results = await asyncio.gather(*eum_tasks)

    # Text ----------------------------------------------------------------------------
    scene_rows = [format_scene_row(cid, search, result, window, (start, end))
                  for search, window, (cid, result)
                  in zip(SCENE_SEARCHES, scene_windows, scene_results, strict=False)
                  if cid is not None]
    sections = []
    eum_rows = [format_days_row(cid, f"{p.label} [{p.key}]", result, (start, end),
                                geometry=geometry_note(bbox, p.sub_satellite_lon))
                for p, (cid, result) in zip(eum_searches, eum_results, strict=False) if cid is not None]
    if eum_rows:
        sections.append("EUMETSAT DATA STORE: products whose footprint includes the bbox, per day "
                        "(full-disk geostationary products always include it: a count shows the "
                        "data exist; the geometry line says whether the area is usable):")
        sections.extend(eum_rows)
        if any("GEOMETRY:" in r for r in eum_rows):
            sections.append(best_geostationary(bbox))
    elif eum_list_id and isinstance(eum_list, dict) and eum_list.get("error"):
        sections.append(f"- [{eum_list_id}] EUMETSAT collection listing FAILED; availability unknown.")
    nasa_rows = [format_days_row(cid, f"{p.label} [{p.key}]", result, (start, end), "granules")
                 for p, (cid, result) in zip(nasa_searches, nasa_results, strict=False) if cid is not None]
    if nasa_rows:
        sections.append("NASA EARTHDATA (CMR): granules intersecting the bbox, per day, with their "
                        "start times = overpass times:")
        sections.extend(nasa_rows)
    cds_rows = [format_cds_row(cid, info) for cid, info in cds_results if cid is not None]
    if cds_rows:
        sections.append("COPERNICUS CLIMATE DATA STORE (reanalysis, gridded, global):")
        sections.extend(cds_rows)
    area = problem.study_area.name if problem.study_area else ""
    return format_inventory(area, bbox, scene_rows, oe_result, oe_id, text, sections)
