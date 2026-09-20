from google.adk.agents.llm_agent import Agent

from agenticeoecosystem.agents.subagents.llm_init import llm
from agenticeoecosystem.agents.subagents.tle_helper import (
    DEFAULT_EO_CONSTELLATION,
    get_tle,
    next_overpasses,
    parse_utc,
    propagate,
)

_TLE_SYSTEM_PROMPT = (
    "You compute satellite orbital position and predict overpasses from TLE "
    "data (Celestrak/Space-Track) via SGP4 propagation. You do not fetch or "
    "analyze imagery, and you cannot identify what satellite produced a past "
    "image — that is out of scope.\n"
    "Tool argument formats: `satellite` is a common platform name (e.g. "
    "'Sentinel-2A', 'Terra') or a NORAD catalog ID as a string; "
    "`timestamp_utc`/`start_utc` are ISO-8601 UTC timestamps.\n"
    "- get_satellite_position / get_positions_for_set: 'where is X (or this "
    "set) right now / at time T' — no target location involved.\n"
    "- get_target_overpasses: 'when will X next pass over this lat/lon' for "
    "one named satellite.\n"
    "- get_acquisition_schedule: the tool for acquisition/monitoring "
    "planning — predicts overpasses of a target lat/lon across a set of "
    "satellites, merged and sorted by time. If the caller doesn't name "
    "specific satellites, omit `satellites` and it defaults to a "
    "wildfire-relevant polar-orbiting EO constellation (Sentinel-2A/B, "
    "Sentinel-3A/B, Terra, Aqua, Suomi NPP, NOAA-20/21, Landsat-8/9). "
    "Geostationary platforms (e.g. Meteosat-11, MTG-I1) have no meaningful "
    "'overpass' — they are fixed relative to Earth — so they are excluded "
    "from this default and from overpass predictions in general; use "
    "get_satellite_position for their current position instead.\n"
    "Always return `accuracy_warning`, `source`, and `tle_age_hours` fields "
    "as-is when present — do not suppress or reword them. A per-satellite "
    "lookup failure inside a set/schedule call is reported in that result's "
    "`errors` list; still return the successful entries rather than failing "
    "the whole call."
)


def get_satellite_position(satellite: str, timestamp_utc: str) -> dict:
    """Get one satellite's lat/lon/altitude at a given UTC timestamp.

    Args:
        satellite: Platform name (e.g. "Sentinel-2A") or NORAD ID string.
        timestamp_utc: ISO-8601 UTC timestamp.

    Returns:
        dict with lat, lon, alt_km, source, tle_age_hours, and an optional
        accuracy_warning.
    """
    tle = get_tle(satellite)
    return propagate(tle, parse_utc(timestamp_utc))


def get_positions_for_set(satellites: list[str], timestamp_utc: str) -> dict:
    """Get lat/lon/altitude for a set of satellites at a given UTC timestamp.

    Args:
        satellites: List of platform names or NORAD ID strings.
        timestamp_utc: ISO-8601 UTC timestamp.

    Returns:
        dict with `positions` (list of per-satellite results) and `errors`
        (list of {satellite, error} for any lookups that failed).
    """
    ts = parse_utc(timestamp_utc)
    positions, errors = [], []
    for sat in satellites:
        try:
            positions.append(propagate(get_tle(sat), ts))
        except Exception as exc:
            errors.append({"satellite": sat, "error": str(exc)})
    return {"positions": positions, "errors": errors}


def get_target_overpasses(
    satellite: str,
    target_lat: float,
    target_lon: float,
    start_utc: str,
    window_hours: float = 24.0,
) -> dict:
    """Predict when one satellite will next pass over a target location.

    Args:
        satellite: Platform name or NORAD ID string.
        target_lat: Target latitude in degrees.
        target_lon: Target longitude in degrees.
        start_utc: ISO-8601 UTC timestamp to start searching from.
        window_hours: How many hours ahead to search.

    Returns:
        dict with the target, window_hours, and a `passes` list of
        {rise_utc, culmination_utc, set_utc, max_elevation_deg}.
    """
    tle = get_tle(satellite)
    passes = next_overpasses(
        tle, target_lat, target_lon, parse_utc(start_utc), window_hours
    )
    return {
        "target": {"lat": target_lat, "lon": target_lon},
        "window_hours": window_hours,
        "passes": passes,
    }


def get_acquisition_schedule(
    target_lat: float,
    target_lon: float,
    start_utc: str,
    window_hours: float = 24.0,
    satellites: list[str] | None = None,
) -> dict:
    """Predict overpasses of a target location across a set of satellites.

    The primary tool for acquisition/monitoring planning: merges and
    time-sorts predicted passes for multiple platforms so the caller can see
    which satellite will be overhead next.

    Args:
        target_lat: Target latitude in degrees.
        target_lon: Target longitude in degrees.
        start_utc: ISO-8601 UTC timestamp to start searching from.
        window_hours: How many hours ahead to search.
        satellites: Platform names/NORAD IDs to check. Defaults to a
            wildfire-relevant EO constellation if omitted.

    Returns:
        dict with the target, window_hours, a time-sorted `schedule` list,
        and `errors` for any satellites that couldn't be resolved.
    """
    sats = satellites if satellites else DEFAULT_EO_CONSTELLATION
    start = parse_utc(start_utc)
    schedule, errors = [], []
    for sat in sats:
        try:
            schedule.extend(
                next_overpasses(get_tle(sat), target_lat, target_lon, start, window_hours)
            )
        except Exception as exc:
            errors.append({"satellite": sat, "error": str(exc)})
    schedule.sort(key=lambda p: p["rise_utc"])
    return {
        "target": {"lat": target_lat, "lon": target_lon},
        "window_hours": window_hours,
        "schedule": schedule,
        "errors": errors,
    }


def _fetch_agent() -> Agent:
    return Agent(
        model=llm,
        name="tle_agent",
        description=(
            "Computes satellite orbital position and predicts upcoming "
            "overpasses of a target location, from TLE data via SGP4 "
            "propagation."
        ),
        instruction=_TLE_SYSTEM_PROMPT,
        tools=[
            get_satellite_position,
            get_positions_for_set,
            get_target_overpasses,
            get_acquisition_schedule,
        ],
        mode="single_turn",
    )


tle_agent = _fetch_agent()
