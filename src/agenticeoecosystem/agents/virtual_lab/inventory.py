"""
inventory.py — The lab's DATA INVENTORY: what data exist for the study area and period.

Built once, right after problem formulation, by plain function calls to public metadata APIs
(no login, no processing credits), run concurrently:
  - Copernicus Data Space scene searches for Sentinel-1/2/3 and Landsat, per acquisition day
    (always: the plan validator checks plan dates against them);
  - DATA DISCOVERY (discovery.py): every provider's collections (Copernicus Data Space STAC,
    EUMETSAT Data Store, NASA Earthdata CMR, Copernicus Climate Data Store) ranked by meaning
    against the problem's data needs, and the shortlist checked for data in the bbox and period;
  - only if discovery found nothing: the fixed core lookups (EUMETSAT geostationary and AVHRR
    product counts, ERA5 coverage).

What is relevant comes from the problem's data needs, the catalogues' metadata and an embedding
model, not from a word list, so problems outside earlier domains get relevant data too.
Every participant sees the same inventory, so availability questions are answered once from
retrieved data instead of by each agent through sub-agent calls.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING, Any

from agenticeoecosystem.servers.cdse_geocode.catalog import ELSEWHERE

if TYPE_CHECKING:
    from .discovery import EmbeddingClient
    from .schemas import ScientificProblem

from .sensor_catalog import (
    GEO_VZA_DEGRADED_DEG,
    GEO_VZA_UNUSABLE_DEG,
    SENSORS,
    bbox_centre,
    geometry_verdict,
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
# Periods longer than this are summarised per calendar month instead of listed per day.
SEASONAL_AFTER_DAYS = 366
CLEAR_CLOUD_PCT = 20   # a day counts as clear at or below this scene cloud cover
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
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


def seasonal_view(problem: ScientificProblem, start: str | None, end: str | None) -> bool:
    """Summarise scenes per calendar month when there is no event to frame them around: a
    generic question, or a period too long to list per day (run of 2026-11-16: a generic
    archaeology question got 'BEFORE / DURING the event' over seven years of daily rows)."""
    if problem.generic:
        return True
    if not (start and end):
        return False
    return (date.fromisoformat(end) - date.fromisoformat(start)).days > SEASONAL_AFTER_DAYS


def _season_text(days: dict[str, dict[str, Any]]) -> str:
    """Per calendar month over all years: days with scenes, median cloud, clear days."""
    by_month: dict[int, list[dict[str, Any]]] = {}
    for day, info in days.items():
        by_month.setdefault(int(day[5:7]), []).append(info)
    parts = []
    for month, name in enumerate(_MONTHS, 1):
        infos = by_month.get(month, [])
        clouds = [median(i["cloud"]) for i in infos if i["cloud"]]
        text = f"{name} {len(infos)}d"
        if clouds:
            clear = sum(c <= CLEAR_CLOUD_PCT for c in clouds)
            text += f" cloud~{median(clouds):.0f}% ({clear} clear)"
        parts.append(text)
    years = sorted({d[:4] for d in days})
    span = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return (f"per calendar month over {span} (days with scenes"
            + (f", median cloud, days <={CLEAR_CLOUD_PCT}% cloud" if any(i["cloud"] for i in days.values())
               else "") + "): " + " | ".join(parts))


def format_scene_row(call_id: str, search: SceneSearch, result: dict[str, Any] | None,
                     window: tuple[str, str], event: tuple[str, str] | None = None,
                     seasonal: bool = False) -> str:
    """One catalogue search; with `event` (start, end) the days are split into
    before / during / after it, so pre- and post-event images are explicit; `seasonal`
    summarises them per calendar month instead."""
    head = f"- [{call_id}] {search.label}, {window[0]}..{window[1]}"
    if not result or result.get("error"):
        error = (result or {}).get("error", "no result")
        return f"{head}: SEARCH FAILED ({error[:120]}). Availability unknown, NOT absent."
    total, listed = result.get("total", 0), result.get("listed", 0)
    if not total:
        return f"{head}: no scenes in the catalogue."
    days = summarise_scenes(result)
    note = "" if listed == total else f" (first {listed} of {total} listed; later days not shown)"
    if seasonal:
        first, last = next(iter(days)), next(reversed(days))
        return (f"{head}: {total} scenes on {len(days)} days{note}, {first} to {last}\n    "
                + _season_text(days))
    lines = []
    for label, group in _event_groups(days, event):
        body = " | ".join(_day_text(d, days[d]) for d in group) if group else "none"
        lines.append(f"{label}: {body}" if label else body)
    return f"{head}: {total} scenes on {len(days)} days{note}\n    " + "\n    ".join(lines)


def format_inventory(problem_area: str, bbox: list[float] | None, scene_rows: list[str],
                     other_sections: list[str] = (), seasonal: bool = False) -> str:
    lines = [f"Study area: {problem_area} bbox {bbox}" if bbox else
             "No study-area bbox, so no scene or product searches were run."]
    if scene_rows:
        lines.append("COPERNICUS DATA SPACE: catalogue scenes intersecting the bbox, "
                     + ("summarised per calendar month (when in the year usable images exist):"
                        if seasonal else "per acquisition day (count, product types, cloud cover, "
                                         "start times):"))
        lines.extend(scene_rows)
    lines.extend(other_sections)
    if other_sections:
        lines.append("NOT in this inventory (its silence about them says nothing): providers other "
                     "than those listed above, e.g. GOES ABI, Himawari AHI, commercial imagery.")
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
    key: str            # EUMETSAT collection id
    label: str
    sub_satellite_lon: float | None = None  # geostationary products: where the satellite sits


# Fixed core lookups, used only when discovery found nothing.
EUMETSAT_CORE: tuple[ProductSearch, ...] = (
    ProductSearch("EO:EUM:DAT:0662", "MTG FCI L1c normal resolution (0 deg, 10 min)",
                  sub_satellite_lon=0.0),
    ProductSearch("EO:EUM:DAT:MSG:HRSEVIRI", "MSG SEVIRI L1.5 (0 deg, 15 min)",
                  sub_satellite_lon=0.0),
    ProductSearch("EO:EUM:DAT:METOP:AVHRRL1", "Metop AVHRR L1B"),
)
CDS_DATASETS = ("reanalysis-era5-single-levels", "reanalysis-era5-land")


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
    """Deprecated: use sensor_catalog.bbox_centre (antimeridian-aware). No callers remain."""
    return bbox_centre(bbox)


def geometry_note(bbox: list[float] | None, sub_satellite_lon: float | None) -> str:
    """Viewing-angle verdict for a geostationary product over the bbox centre ('' if n/a)."""
    if not bbox or sub_satellite_lon is None:
        return ""
    vza = geostationary_vza_deg(*bbox_centre(bbox), sub_satellite_lon)
    where = f"viewing zenith angle {vza:.0f} deg from the satellite at {sub_satellite_lon:+.1f} deg"
    verdict = geometry_verdict(vza)
    if verdict == "unusable":
        return (f"UNUSABLE GEOMETRY: {where} (> {GEO_VZA_UNUSABLE_DEG:.0f}): the data exist but "
                "this area is at the disk edge; do not plan quantitative use.")
    if verdict == "degraded":
        return (f"DEGRADED GEOMETRY: {where} (> {GEO_VZA_DEGRADED_DEG:.0f}): large pixels and long "
                "slant paths; usable with caution.")
    return f"Geometry OK: {where}."


def best_geostationary(bbox: list[float]) -> str:
    """The catalog's geostationary imager with the lowest viewing angle over the bbox."""
    lat, lon = bbox_centre(bbox)
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


async def build_inventory(problem: ScientificProblem, catalogs: Catalogs, call: Call,
                          pad_days: int, embedder: EmbeddingClient | None = None) -> str:
    """Run the scene searches and the data discovery concurrently (and the core lookups if
    discovery found nothing); return the DATA INVENTORY text shown to all participants."""
    from . import discovery

    bbox = list(problem.study_area.bbox) if problem.study_area and problem.study_area.bbox else None
    period = problem.time_period
    start, end = ((period.start[:10], period.end[:10])
                  if period and valid_period(period.start, period.end) else (None, None))
    searchable = bool(bbox and start)
    cdse, eum, cds = catalogs.cdse, catalogs.eumetsat, catalogs.cds

    scenes, scene_windows = [], []
    if cdse and searchable and hasattr(cdse, "search_scenes"):
        for search in SCENE_SEARCHES:
            window = search_window(start, end, search, pad_days)
            args = {"collection": search.collection, "bbox": bbox, "start_date": window[0],
                    "end_date": window[1], "product_type": search.product_type, "limit": SCENE_LIMIT}
            scenes.append(call("search_cdse_catalog", args, lambda a=args: cdse.search_scenes(**a)))
            scene_windows.append(window)
    found, *scene_results = await asyncio.gather(
        discovery.discover(problem, catalogs, call, embedder), *scenes)

    def geometry(hit) -> str:
        if hit.candidate.provider != "eumetsat":
            return ""
        return geometry_note(bbox, geostationary_lon(hit.candidate.title))

    sections = [discovery.render(found, geometry)]
    if not found.hits:
        sections += await _core_lookups(eum, cds, bbox, start, end, searchable, call)
    seasonal = seasonal_view(problem, start, end)
    scene_rows = [format_scene_row(cid, search, result, window, None if seasonal else (start, end),
                                   seasonal)
                  for search, window, (cid, result)
                  in zip(SCENE_SEARCHES, scene_windows, scene_results, strict=False)
                  if cid is not None]
    area = problem.study_area.name if problem.study_area else ""
    return format_inventory(area, bbox, scene_rows, sections, seasonal)


async def _core_lookups(eum, cds, bbox, start, end, searchable: bool, call: Call) -> list[str]:
    """Fallback when discovery found nothing: EUMETSAT core product counts and ERA5 coverage."""
    eum_searches = list(EUMETSAT_CORE) if eum and searchable else []
    eum_tasks = [call("count_eumetsat_products", a, lambda a=a: eum.search_products(**a))
                 for a in ({"collection_id": p.key, "bbox": bbox, "start_date": start, "end_date": end}
                           for p in eum_searches)]
    cds_tasks = [call("cds_dataset_info", a, lambda a=a: cds.dataset_info(**a))
                 for a in ({"dataset_id": d, "start_date": start, "end_date": end}
                           for d in (CDS_DATASETS if cds else ()))]
    results = await asyncio.gather(*eum_tasks, *cds_tasks)
    eum_results, cds_results = results[:len(eum_tasks)], results[len(eum_tasks):]
    sections = []
    eum_rows = [format_days_row(cid, f"{p.label} [{p.key}]", result, (start, end),
                                geometry=geometry_note(bbox, p.sub_satellite_lon))
                for p, (cid, result) in zip(eum_searches, eum_results, strict=False) if cid is not None]
    if eum_rows:
        sections.append("EUMETSAT DATA STORE (core products; discovery found nothing): products "
                        "whose footprint includes the bbox, per day:")
        sections.extend(eum_rows)
        if any("GEOMETRY:" in r for r in eum_rows):
            sections.append(best_geostationary(bbox))
    cds_rows = [format_cds_row(cid, info) for cid, info in cds_results if cid is not None]
    if cds_rows:
        sections.append("COPERNICUS CLIMATE DATA STORE (core reanalyses):")
        sections.extend(cds_rows)
    return sections


# ------------------------------------------------------------------------- dataset registry

STATUS_RANK = {"verified": 3, "partial": 2, "unknown": 1, "absent": 0}


@dataclass(frozen=True)
class RegistryEntry:
    """One collection a catalogue lookup of this run reported on (a view of the ledger:
    LabLedger.tool_calls entries carry "data" rows, see ledger.data_records)."""
    id: str                 # "tc-0037", or "tc-0052/NISAR_L2_GSLC_V1" for one of many
    provider: str
    collection: str
    title: str
    status: str             # verified | partial | absent | unknown
    availability: str
    found_by: str           # "inventory" or the agent that searched
    step: str


def data_registry(tool_calls: list[dict[str, Any]]) -> dict[str, RegistryEntry]:
    """Every dataset this run's catalogue lookups reported on, by registry id. Built from
    the ledger on demand, so it always reflects the searches made so far."""
    registry: dict[str, RegistryEntry] = {}
    for call in tool_calls:
        if call.get("status") != "ok":
            continue
        for row in call.get("data") or []:
            rid = f"{call['id']}/{row['key']}" if row.get("key") else call["id"]
            registry[rid] = RegistryEntry(
                id=rid, provider=row.get("provider", ""), collection=row.get("collection", ""),
                title=row.get("title", ""), status=row.get("status", "unknown"),
                availability=row.get("availability", ""),
                found_by="inventory" if call.get("inventory") else str(call.get("agent") or ""),
                step=str(call.get("step") or ""))
    return registry


def resolve(registry: dict[str, RegistryEntry], source_id: str) -> list[RegistryEntry]:
    """Entries a cited id refers to: the exact id, or every collection of a cited
    multi-collection call ("tc-0052" for "tc-0052/<id>" rows)."""
    if source_id in registry:
        return [registry[source_id]]
    return [e for rid, e in registry.items() if rid.startswith(f"{source_id}/")]


def best_status(entries: list[RegistryEntry]) -> str | None:
    return max((e.status for e in entries), key=STATUS_RANK.get, default=None)


def format_registry(registry: dict[str, RegistryEntry], max_rows: int = 60) -> str:
    """DATA REGISTRY section: datasets found by the agents' own searches (the DATA INVENTORY
    rows are registry entries too and are not repeated). Absent ones are listed by id only."""
    extra = [e for e in registry.values() if e.found_by != "inventory"]
    if not extra:
        return ""
    shown = sorted((e for e in extra if e.status != "absent"),
                   key=lambda e: -STATUS_RANK.get(e.status, 0))[:max_rows]
    lines = [f"- [{e.id}] {e.provider} · {e.collection}" + (f" — {e.title}" if e.title else "")
             + f": {e.status.upper()}; {e.availability} (searched by {e.found_by})" for e in shown]
    absent = [e.id for e in extra if e.status == "absent"]
    if absent:
        lines.append("- NONE found in the bbox and period: " + ", ".join(absent))
    if len(extra) - len(absent) > len(shown):
        lines.append(f"- ... {len(extra) - len(absent) - len(shown)} more")
    return "\n".join(lines)
