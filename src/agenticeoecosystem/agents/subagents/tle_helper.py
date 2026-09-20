"""TLE fetching (Celestrak/Space-Track), caching, and SGP4 propagation.

Pure logic module, no ADK imports — used by sub_agent_tle.py.
"""
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from pyorbital.orbital import Orbital
from skyfield.api import EarthSatellite, load, wgs84

from agenticeoecosystem.definitions import ROOT_DIR

logger = logging.getLogger(__name__)

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_QUERY_URL = (
    "https://www.space-track.org/basicspacedata/query/class/tle_latest/"
    "NORAD_CAT_ID/{norad_id}/ORDINAL/1/format/tle"
)

TLE_CACHE_DIR = ROOT_DIR / "output" / "tle"
TLE_MAX_AGE_HOURS = 24
# SGP4 accuracy degrades noticeably once the TLE epoch is this far from the
# requested timestamp; surfaced to the caller rather than silently ignored.
ACCURACY_WARNING_HOURS = 120

NORAD_ID_MAP: dict[str, int] = {
    "sentinel-2a": 40697,
    "sentinel-2b": 42063,
    "sentinel-2c": 60418,
    "sentinel-3a": 41335,
    "sentinel-3b": 43437,
    "terra": 25994,
    "aqua": 27424,
    "suomi npp": 37849,
    "noaa-20": 43013,
    "noaa-21": 54234,
    "meteosat-11": 40732,
    "msg": 40732,
    "msg / seviri": 40732,
    "msg/seviri": 40732,
    "seviri": 40732,
    "mtg-i1": 54743,
    "mtg": 54743,
    "mtg / fci": 54743,
    "mtg/fci": 54743,
    "fci": 54743,
    "landsat-8": 39084,
    "landsat-9": 49260,
    "iss": 25544,
}

# Wildfire-relevant EO platforms, used as the default candidate set for
# get_acquisition_schedule when the caller doesn't name specific satellites.
# Geostationary platforms (meteosat-11, mtg-i1) are deliberately excluded:
# pyorbital's SGP4 propagator doesn't support "deep space" (near-24h period)
# orbits, and overpass timing isn't a meaningful concept for a satellite
# that's fixed relative to Earth anyway — query their position directly via
# get_satellite_position instead.
DEFAULT_EO_CONSTELLATION: list[str] = [
    "sentinel-2a", "sentinel-2b", "sentinel-3a", "sentinel-3b",
    "terra", "aqua", "suomi npp", "noaa-20", "noaa-21",
    "landsat-8", "landsat-9",
]

_spacetrack_session: requests.Session | None = None
_ts = load.timescale()  # skyfield time scale, built once and reused


def _parse_tle_epoch(tle_line1: str) -> datetime:
    """Extract the epoch from TLE line 1 (columns 19-32: YYDDD.DDDDDDDD)
    without needing a propagator library — kept independent of both
    pyorbital and skyfield so caching doesn't depend on either."""
    epoch_field = tle_line1[18:32]
    year_two = int(epoch_field[0:2])
    day_frac = float(epoch_field[2:])
    year = 2000 + year_two if year_two < 57 else 1900 + year_two
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1)


def parse_utc(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp (with or without a trailing 'Z') to an
    aware UTC datetime."""
    s = timestamp.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_norad_id(satellite: str) -> int:
    s = satellite.strip()
    if s.isdigit():
        return int(s)
    key = s.lower()
    if key in NORAD_ID_MAP:
        return NORAD_ID_MAP[key]
    resp = requests.get(
        CELESTRAK_GP_URL, params={"NAME": s, "FORMAT": "json"}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"No satellite found matching name '{satellite}'.")
    return int(data[0]["NORAD_CAT_ID"])


def fetch_tle_celestrak(norad_id: int) -> dict | None:
    resp = requests.get(
        CELESTRAK_GP_URL, params={"CATNR": norad_id, "FORMAT": "TLE"}, timeout=15
    )
    if resp.status_code != 200 or not resp.text.strip():
        return None
    lines = [line for line in resp.text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    name, line1, line2 = lines[0].strip(), lines[1].strip(), lines[2].strip()
    return {
        "norad_id": norad_id, "name": name,
        "tle_line1": line1, "tle_line2": line2, "source": "celestrak",
    }


def _get_spacetrack_session() -> requests.Session | None:
    global _spacetrack_session
    if _spacetrack_session is not None:
        return _spacetrack_session
    load_dotenv(ROOT_DIR / ".env")
    user = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASSWORD")
    if not user or not password:
        return None
    session = requests.Session()
    resp = session.post(
        SPACETRACK_LOGIN_URL,
        data={"identity": user, "password": password},
        timeout=15,
    )
    if resp.status_code != 200:
        logger.warning("Space-Track login failed with status %s.", resp.status_code)
        return None
    _spacetrack_session = session
    return session


def fetch_tle_spacetrack(norad_id: int) -> dict | None:
    session = _get_spacetrack_session()
    if session is None:
        return None
    resp = session.get(SPACETRACK_QUERY_URL.format(norad_id=norad_id), timeout=15)
    if resp.status_code != 200 or not resp.text.strip():
        return None
    lines = [line for line in resp.text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return {
        "norad_id": norad_id, "name": f"NORAD {norad_id}",
        "tle_line1": lines[0].strip(), "tle_line2": lines[1].strip(),
        "source": "spacetrack",
    }


def get_tle(satellite: str) -> dict:
    """Cache-first TLE lookup: Celestrak first, Space-Track fallback."""
    norad_id = resolve_norad_id(satellite)
    cache_path = TLE_CACHE_DIR / f"{norad_id}.json"
    now = datetime.now(timezone.utc)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (now - fetched_at).total_seconds() < TLE_MAX_AGE_HOURS * 3600:
                return cached
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt/old cache entry, fall through to refetch

    tle = fetch_tle_celestrak(norad_id) or fetch_tle_spacetrack(norad_id)
    if tle is None:
        raise ValueError(
            f"No TLE data available for '{satellite}' (NORAD {norad_id}) "
            "from Celestrak or Space-Track."
        )

    tle["epoch"] = _parse_tle_epoch(tle["tle_line1"]).isoformat()
    tle["fetched_at"] = now.isoformat()

    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(tle, indent=2))
    return tle


def propagate(tle: dict, timestamp_utc: datetime) -> dict:
    """Position at a given UTC timestamp via skyfield/sgp4 — handles both
    near-Earth (LEO/MEO) and deep-space (geostationary) orbits uniformly,
    unlike pyorbital's SGP4-only implementation."""
    sat = EarthSatellite(tle["tle_line1"], tle["tle_line2"], tle["name"], _ts)
    t = _ts.from_datetime(timestamp_utc)
    subpoint = wgs84.subpoint(sat.at(t))

    epoch = datetime.fromisoformat(tle["epoch"])
    age_hours = abs((timestamp_utc - epoch).total_seconds() / 3600)

    result = {
        "satellite": tle["name"],
        "norad_id": tle["norad_id"],
        "timestamp_utc": timestamp_utc.isoformat(),
        "lat": round(float(subpoint.latitude.degrees), 4),
        "lon": round(float(subpoint.longitude.degrees), 4),
        "alt_km": round(float(subpoint.elevation.km), 2),
        "source": tle["source"],
        "tle_age_hours": round(age_hours, 1),
    }
    if age_hours > ACCURACY_WARNING_HOURS:
        result["accuracy_warning"] = (
            f"TLE epoch is {age_hours:.0f}h from the requested timestamp; "
            "SGP4 accuracy degrades noticeably beyond ~5 days."
        )
    return result


def next_overpasses(
    tle: dict,
    target_lat: float,
    target_lon: float,
    start_utc: datetime,
    window_hours: float,
    target_alt_m: float = 0.0,
    horizon_deg: float = 0.0,
) -> list[dict]:
    try:
        orb = Orbital(tle["name"], line1=tle["tle_line1"], line2=tle["tle_line2"])
    except NotImplementedError:
        raise ValueError(
            f"'{tle['name']}' is a geostationary/deep-space satellite; it "
            "has no periodic rise/set and overpass prediction isn't "
            "supported. Use get_satellite_position for its current "
            "position instead."
        )
    naive_start = start_utc.replace(tzinfo=None)
    target_alt_km = target_alt_m / 1000.0

    # pyorbital's get_next_passes does `range(length * 60)` internally, which
    # requires `length` to be an actual `int` (range() rejects a float even
    # when its value is whole, e.g. 24.0) — round up so a fractional-hour
    # window still gets fully covered rather than truncated.
    passes = orb.get_next_passes(
        naive_start, math.ceil(window_hours), target_lon, target_lat,
        target_alt_km, horizon=horizon_deg,
    )

    results = []
    for rise, fall, max_elev_time in passes:
        _, elevation = orb.get_observer_look(
            max_elev_time, target_lon, target_lat, target_alt_km
        )
        results.append({
            "satellite": tle["name"],
            "norad_id": tle["norad_id"],
            "rise_utc": rise.replace(tzinfo=timezone.utc).isoformat(),
            "culmination_utc": max_elev_time.replace(tzinfo=timezone.utc).isoformat(),
            "set_utc": fall.replace(tzinfo=timezone.utc).isoformat(),
            "max_elevation_deg": round(float(elevation), 1),
        })
    return sorted(results, key=lambda p: p["rise_utc"])
