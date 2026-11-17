"""
sensor_catalog.py — Deterministic EO sensor metadata for Virtual Lab validation.

This is the lab's "dataset metadata" grounding source: orbit geometry,
observing constraints, and which existing sub-agent can actually fetch each
dataset. Validation checks run against this table, not against LLM
recall. Wildfire capability scores are copied from SATELLITE_CAPABILITY_MATRIX
in agents/multi_agent_coordination.py (itself transcribed from
data/visual_tasks/context/knowledge.tex); that module can't be imported here
because importing it builds a Runner.

There are no mission dates here: whether data exist for the study area and period
is checked against this run's catalogue results (validation.check_grounding).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

EARTH_RADIUS_KM = 6371.0
GEO_ORBIT_RADIUS_KM = 42164.0

# Viewing zenith angle limits for geostationary imagers: beyond ~70 deg pixel
# footprints and path lengths degrade retrievals; beyond ~80 deg the scene is
# effectively at the disk edge and not usable for quantitative work.
GEO_VZA_DEGRADED_DEG = 70.0
GEO_VZA_UNUSABLE_DEG = 80.0


@dataclass(frozen=True)
class SensorInfo:
    key: str
    aliases: tuple[str, ...]
    orbit: str                       # "polar" | "geostationary" | "reanalysis"
    sub_satellite_lon: float | None = None
    resolution_m: float | None = None
    revisit: str = ""
    day_only: bool = False           # reflective-only: no night observations
    all_weather: bool = False        # penetrates cloud/smoke (SAR)
    thermal: bool = False            # has thermal-IR channels
    sub_agents: tuple[str, ...] = ()
    wildfire_scores: dict[str, int] = field(default_factory=dict)  # AF/FRP/SMK/BA/REC, 0-100
    notes: str = ""


def _scores(af: int, frp: int, smk: int, ba: int, rec: int) -> dict[str, int]:
    return {"AF": af, "FRP": frp, "SMK": smk, "BA": ba, "REC": rec}


# Order matters for alias matching: more specific entries first.
SENSORS: tuple[SensorInfo, ...] = (
    SensorInfo("Sentinel-3 OLCI", ("olci",), "polar", resolution_m=300,
               revisit="~1-2 days", day_only=True, sub_agents=("cdse_agent", "eumetsat_agent"),
               wildfire_scores=_scores(82, 83, 56, 51, 29),
               notes="Scores are for SLSTR/OLCI combined in knowledge.tex."),
    SensorInfo("Sentinel-3 SLSTR", ("slstr", "sentinel3"), "polar",
               resolution_m=1000, revisit="~1 day", thermal=True,
               sub_agents=("cdse_agent", "eumetsat_agent"),
               wildfire_scores=_scores(82, 83, 56, 51, 29),
               notes="Thermal/fire channels at 1 km."),
    SensorInfo("Sentinel-5P TROPOMI", ("tropomi", "sentinel5p", "s5p"), "polar",
               resolution_m=5500, revisit="daily", day_only=True, sub_agents=("cdse_agent",),
               wildfire_scores=_scores(12, 10, 84, 18, 41)),
    SensorInfo("Sentinel-2 MSI", ("sentinel2", "s2l2a", "s2l1c", "msi"), "polar",
               resolution_m=10, revisit="~5 days (2-3 with three units)", day_only=True,
               sub_agents=("cdse_agent", "gee_agent"), wildfire_scores=_scores(65, 46, 73, 92, 94),
               notes="SWIR (B11/B12) can flag very hot fires by day; no thermal-IR band."),
    SensorInfo("Sentinel-1 C-SAR", ("sentinel1", "s1grd", "csar", "sar"), "polar",
               resolution_m=10, revisit="6-12 days", all_weather=True, sub_agents=("cdse_agent", "gee_agent"),
               wildfire_scores=_scores(26, 20, 8, 79, 81)),
    SensorInfo("Landsat 8/9 OLI-TIRS", ("landsat", "oli", "tirs"), "polar",
               resolution_m=30, revisit="8 days (combined)", day_only=True, thermal=True,
               sub_agents=("cdse_agent", "gee_agent"), wildfire_scores=_scores(70, 63, 71, 93, 95),
               notes="CDSE provides L1/TOA only; surface reflectance (L2) via GEE."),
    SensorInfo("VIIRS", ("viirs", "suomi", "npp", "noaa20", "noaa21", "jpss"), "polar",
               resolution_m=375, revisit="~12 h per platform (day + night)",
               thermal=True, sub_agents=("gee_agent",), wildfire_scores=_scores(87, 87, 67, 58, 30)),
    SensorInfo("MODIS", ("modis", "mcd64", "mod14", "myd14"), "polar",
               resolution_m=250, revisit="~12 h per platform (day + night)",
               thermal=True, sub_agents=("gee_agent",), wildfire_scores=_scores(81, 81, 61, 52, 28),
               notes="Terra/Aqua are in end-of-mission orbit drift; check data continuity for recent dates."),
    SensorInfo("MTG FCI", ("fci", "mtg"), "geostationary", sub_satellite_lon=0.0,
               resolution_m=2000, revisit="10 min full disk", thermal=True,
               sub_agents=("eumetsat_agent",), wildfire_scores=_scores(83, 84, 70, 28, 16)),
    SensorInfo("MSG SEVIRI", ("seviri", "msg", "meteosat"), "geostationary",
               sub_satellite_lon=0.0, resolution_m=3000, revisit="15 min full disk", thermal=True,
               sub_agents=("eumetsat_agent",), wildfire_scores=_scores(70, 66, 64, 25, 16)),
    SensorInfo("GOES-East ABI", ("goes", "abi"), "geostationary",
               sub_satellite_lon=-75.2, resolution_m=2000, revisit="10 min full disk", thermal=True,
               wildfire_scores=_scores(74, 71, 69, 25, 16),
               notes="No sub-agent in this system can fetch GOES data."),
    SensorInfo("ERA5", ("era5",), "reanalysis", resolution_m=31000,
               revisit="hourly", sub_agents=("climate_agent",),
               notes="Model reanalysis, not an observation; ~5-day latency."),
)


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def lookup(name: str) -> SensorInfo | None:
    """Match a free-text dataset name (e.g. 'Sentinel-2 L2A dNBR') to a catalog entry."""
    norm = _normalise(name)
    for sensor in SENSORS:
        if _normalise(sensor.key) == norm or any(alias in norm for alias in sensor.aliases):
            return sensor
    return None


def sensors_in_text(text: str) -> list[SensorInfo]:
    """Every catalog sensor named in free text. Aliases must match whole words or
    joined runs of up to 3 words ('Sentinel-2' -> 'sentinel2'), so 'sar' does not
    match 'necessary' and 'msg' does not match 'message'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = set(words)
    for n in (2, 3):
        grams |= {"".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return [s for s in SENSORS
            if _normalise(s.key) in grams or any(alias in grams for alias in s.aliases)]


def check_bbox(bbox: list[float] | None, allow_antimeridian: bool = True) -> str | None:
    """Why bbox [west, south, east, north] (degrees) is invalid, or None. west > east means
    the box crosses the antimeridian, unless allow_antimeridian is False."""
    if bbox is None or len(bbox) != 4:
        return "bbox must be [west, south, east, north]."
    west, south, east, north = bbox
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90):
        return "bbox must be in degrees with south < north."
    if west == east or (west > east and not allow_antimeridian):
        return "bbox must have west < east (boxes crossing the antimeridian are not supported here)."
    return None


def bbox_centre(bbox: list[float]) -> tuple[float, float]:
    """(lat, lon) of the centre of bbox [west, south, east, north], antimeridian-aware."""
    west, south, east, north = bbox
    if west > east:  # crosses the antimeridian
        east += 360
    lon = (west + east) / 2
    return (south + north) / 2, (lon - 360 if lon > 180 else lon)


def geometry_verdict(vza_deg: float) -> str:
    """'unusable', 'degraded' or 'ok' for a geostationary viewing zenith angle."""
    if vza_deg > GEO_VZA_UNUSABLE_DEG:
        return "unusable"
    return "degraded" if vza_deg > GEO_VZA_DEGRADED_DEG else "ok"


def geostationary_vza_deg(lat: float, lon: float, sub_satellite_lon: float) -> float:
    """Viewing zenith angle (deg) of a geostationary satellite from a ground point."""
    phi = math.radians(lat)
    dlon = math.radians(lon - sub_satellite_lon)
    cos_gamma = math.cos(phi) * math.cos(dlon)
    slant = math.sqrt(EARTH_RADIUS_KM**2 + GEO_ORBIT_RADIUS_KM**2
                      - 2 * EARTH_RADIUS_KM * GEO_ORBIT_RADIUS_KM * cos_gamma)
    cos_vza = (GEO_ORBIT_RADIUS_KM * cos_gamma - EARTH_RADIUS_KM) / slant
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_vza))))


def format_catalog() -> str:
    """Compact table for participant prompts."""
    rows = []
    for s in SENSORS:
        flags = [f for f, on in (("day-only", s.day_only), ("all-weather", s.all_weather),
                                 ("thermal", s.thermal)) if on]
        geo = f" @ {s.sub_satellite_lon:+.1f} deg" if s.sub_satellite_lon is not None else ""
        access = ", ".join(s.sub_agents) or "NO ACCESS in this system"
        scores = " ".join(f"{k}={v}" for k, v in s.wildfire_scores.items())
        rows.append(
            f"- {s.key}: {s.orbit}{geo}, "
            f"~{s.resolution_m:g} m, revisit {s.revisit}; {', '.join(flags) or 'reflective/other'}; "
            f"fetch via {access}" + (f"; wildfire capability {scores}" if scores else "")
            + (f". Note: {s.notes}" if s.notes else "")
        )
    return "\n".join(rows)
