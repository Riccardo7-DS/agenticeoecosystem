#!/usr/bin/env python
"""
run_acquisition_coverage_emsr859.py — Fetch-correctness test for the TLE
sub-agent's get_acquisition_schedule tool, isolated from the VLM/image
pipeline that run_visual_decision_wildfire.py exercises.

Uses a REAL anchor instead of a synthetic one: the AOI of wildfire event
EMSR859_1 (Copernicus EMS activation covered by the ImpactMesh-Fire dataset,
data/ImpactMesh-Fire/data/S2L2A/EMSR859_1_*), south-central Chile. AOI
centroid/bbox were derived from that dataset's own S2L2A tile metadata
(.zattrs: center_lat/center_lon, plus the UTM-32718 x/y extent across all 30
tiles belonging to EMSR859_1) — not looked up externally.

Restricts sub-agent loading to tle only (AGENTICEO_SUBAGENTS=tle) before
importing the orchestrator, so no VLM/image step is involved at all: this
run tests ONLY whether get_acquisition_schedule, called through the
orchestrator, correctly restricts itself to the platforms this system
actually holds TLE data for (NORAD_ID_MAP in tle_helper.py) and correctly
identifies which of them pass closest to overhead (max elevation) — i.e.
"maximum coverage" of the AOI — inside a fixed real-world time window.

Logs go to logs/acquisition_coverage_emsr859_<timestamp>.log (via the
project's setup_logging) plus a JSON result file with the same timestamp.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("AGENTICEO_SUBAGENTS", "tle")

from dotenv import load_dotenv
load_dotenv()

import logging
from agenticeoecosystem.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# --- EMSR859_1 AOI, derived from the dataset's own tile metadata ----------
# Centroid of the T18HYD tile nearest the middle of the 30-tile AOI extent
# (data/ImpactMesh-Fire/data/S2L2A/EMSR859_1_18HYD_x709105_y5867715_S2L2A.zarr.zip's
# .zattrs: center_lat/center_lon). Full AOI bbox (all 30 tiles, UTM 32718
# easting 701425-714225 / northing 5857475-5877955) spans roughly
# lat -37.41..-37.22, lon -72.73..-72.59 (~21 km x 21 km).
AOI_LAT = -37.3249
AOI_LON = -72.6256

WINDOW_START_UTC = "2026-01-17T00:00:00Z"
WINDOW_HOURS = 48.0  # covers 17 Jan and 18 Jan fully

# Every non-geostationary platform this system holds TLE data for
# (tle_helper.NORAD_ID_MAP minus ISS, which isn't an EO platform, and minus
# the geostationary entries, which have no overpass semantics). Includes
# sentinel-2c explicitly since it's in NORAD_ID_MAP but NOT in
# DEFAULT_EO_CONSTELLATION, and the EMSR859_1 tile metadata itself shows
# Sentinel-2C actually imaged this exact AOI.
KNOWN_OVERPASS_SATELLITES = [
    "Sentinel-2A", "Sentinel-2B", "Sentinel-2C",
    "Sentinel-3A", "Sentinel-3B",
    "Terra", "Aqua", "Suomi NPP",
    "NOAA-20", "NOAA-21",
    "Landsat-8", "Landsat-9",
]
KNOWN_GEO_SATELLITES = ["Meteosat-11", "MTG-I1"]  # position-only, no overpass

_QUERY_TEMPLATE = """\
This is a fetch-correctness test of the TLE sub-agent, not an image-analysis \
task — no imagery is attached, and no VLM step is required. Do not produce a \
FEATURES ANALYSIS section.

TARGET: the AOI of wildfire event EMSR859_1 (south-central Chile), centered \
at latitude {lat}, longitude {lon}, covering an area of roughly 21 km x 21 km \
around that point.

WINDOW: from {start_utc} to 48 hours later (i.e. through the end of 18 \
January) — window_hours={window_hours}.

SATELLITE SCOPE — this is the important constraint: this system has TLE data \
for exactly these platforms, and no others. You must restrict your entire \
analysis to this closed list; do not attempt to look up, guess, add, or \
fetch orbital data for any satellite not named here, even if you believe it \
would be relevant to wildfire monitoring:
Non-geostationary (has a meaningful overpass over the AOI):
{overpass_list}
Geostationary (fixed relative to Earth, no overpass — excluded from this
test's coverage question, do not query them):
{geo_list}

TASK:
1. Call the TLE sub-agent's `get_acquisition_schedule` tool EXACTLY ONCE, \
passing `satellites` as the full non-geostationary list above verbatim (all \
12 platforms in one call — do not split it into per-satellite \
get_target_overpasses calls), target_lat={lat}, target_lon={lon}, \
start_utc="{start_utc}", window_hours={window_hours}. You MUST actually make \
this tool call and read its result before answering — never write \
"I will now fetch..." as a substitute for calling it.
2. From the returned schedule, report every predicted pass in the window as \
a time-sorted table: satellite, rise_utc, culmination_utc, set_utc, \
max_elevation_deg. If a listed satellite has ZERO passes in the window, \
state that explicitly rather than omitting it.
3. Identify MAXIMUM COVERAGE: the satellite(s) whose pass comes closest to \
directly overhead the AOI — i.e. the highest max_elevation_deg values \
(closest to 90 degrees) — since those give the best near-nadir view of the \
AOI with the least off-angle distortion. Rank all passes by max_elevation_deg \
descending and call out the single best pass, plus any other pass above 60 \
degrees elevation as also offering strong coverage.
4. State plainly which satellites in the allowed list pass "exactly above" \
(i.e. near-zenith, informally >= 80 degrees max elevation) the EMSR859_1 AOI \
in this window, if any — this is the test's pass/fail signal for whether the \
fetching is working correctly and returning physically plausible geometry.
"""


def _build_query() -> str:
    return _QUERY_TEMPLATE.format(
        lat=AOI_LAT,
        lon=AOI_LON,
        start_utc=WINDOW_START_UTC,
        window_hours=WINDOW_HOURS,
        overpass_list="\n".join(f"- {s}" for s in KNOWN_OVERPASS_SATELLITES),
        geo_list="\n".join(f"- {s}" for s in KNOWN_GEO_SATELLITES),
    )


async def _run(m, out_path: Path) -> None:
    query = _build_query()
    logger.info("EMSR859_1 AOI: (%.4f, %.4f)", AOI_LAT, AOI_LON)
    logger.info(
        "Window: %s + %sh", WINDOW_START_UTC, WINDOW_HOURS
    )
    logger.info("Query:\n%s", query)

    result = {
        "case": "EMSR859_1",
        "aoi_lat": AOI_LAT,
        "aoi_lon": AOI_LON,
        "window_start_utc": WINDOW_START_UTC,
        "window_hours": WINDOW_HOURS,
        "allowed_satellites": KNOWN_OVERPASS_SATELLITES,
    }
    try:
        raw = await m.get_ans_tool_path(query)
        result.update(json.loads(raw))
    except Exception as e:
        logger.error("FAILED: %s", e)
        result["error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Result written to %s", out_path)

    final_answer = result.get("final_answer")
    if final_answer:
        logger.info("\n%s\n%s\n%s", "=" * 80, final_answer, "=" * 80)
    if result.get("tools_called"):
        logger.info(
            "Tools called: %s",
            [t["function_name"] for t in result["tools_called"]],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    setup_logging(log_file="acquisition_coverage_emsr859")

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"acquisition_coverage_emsr859_result_{out_timestamp}.json"

    # Imported here (synchronous context, after AGENTICEO_SUBAGENTS is set and
    # logging is configured) because the module itself calls asyncio.run() at
    # import time to set up its session — nesting that inside our own running
    # event loop would raise.
    from agenticeoecosystem.agents import multi_agent_coordination as m

    asyncio.run(_run(m, out_path))


if __name__ == "__main__":
    main()
