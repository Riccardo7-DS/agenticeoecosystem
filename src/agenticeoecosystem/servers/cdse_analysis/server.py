"""
MCP server exposing the Virtual Lab's fixed burned-area analyses on Copernicus data
(agents/virtual_lab/analyses.py), computed on Sentinel Hub at the Copernicus Data Space.

  optical_conditions     cloud / smoke per Sentinel-2 scene; picks the clearest pre/post scenes
  s2_dnbr_burned_area    Sentinel-2 dNBR burned area
  s1_backscatter_change  Sentinel-1 VH backscatter-change burned area (unaffected by cloud/smoke)
  s1_s2_agreement        overlap of the two burned-area maps, with a quicklook PNG

Each tool is one call. Results and maps of a study (bbox, period, padding) are saved in
<output>/cdse_analysis/<study>/, so a later call reuses them: s1_s2_agreement after the two
burned-area calls downloads nothing. A tool runs the analyses it needs that have no saved
result yet. The lab itself calls the same functions in-process.

Needs CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (Sentinel Hub processing quota, not openEO credits).
Bounding boxes are [west, south, east, north] in WGS-84 degrees, as on cdse_geocode.

Run with:
    python server.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from agenticeoecosystem.agents.virtual_lab.analyses import (
    ANALYSES,
    DEPENDS,
    AnalysisContext,
    AnalysisStore,
    SentinelHubClient,
)
from agenticeoecosystem.agents.virtual_lab.sensor_catalog import check_bbox
from agenticeoecosystem.definitions import ROOT_DIR
from agenticeoecosystem.utils.output_paths import server_output_dir

load_dotenv(dotenv_path=ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, stream=sys.stderr,   # stdout is the MCP stream
                    format="%(asctime)s - CDSE-ANALYSIS-MCP - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastMCP("cdse-analysis-mcp")

OUTPUT_DIR = server_output_dir("cdse_analysis", "CDSE_ANALYSIS_OUTPUT_DIR")
MAX_PAD_DAYS = 60
_client: SentinelHubClient | None = None


def _check(bbox: list[float], start_date: str, end_date: str, pad_days: int) -> str | None:
    """Why the inputs are invalid, or None."""
    if error := check_bbox(bbox, allow_antimeridian=False):   # the analysis grid needs west < east
        return error
    try:
        start, end = date.fromisoformat(start_date[:10]), date.fromisoformat(end_date[:10])
    except ValueError:
        return "start_date and end_date must be YYYY-MM-DD."
    if start > end:
        return "start_date is after end_date."
    if not 0 <= pad_days <= MAX_PAD_DAYS:
        return f"pad_days must be between 0 and {MAX_PAD_DAYS}."
    return None


async def _run(name: str, bbox: list[float], start_date: str, end_date: str, pad_days: int,
               refresh: bool) -> dict[str, Any]:
    global _client
    if error := _check(bbox, start_date, end_date, pad_days):
        return {"status": "failed", "error": error}
    if _client is None and (_client := SentinelHubClient.from_env()) is None:
        return {"status": "failed", "error": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set."}
    ctx = AnalysisContext(bbox=list(bbox), start=start_date[:10], end=end_date[:10], pad_days=pad_days)
    return {"analysis": name, **await AnalysisStore(OUTPUT_DIR).run(name, ctx, _client, refresh)}


_ARGS = """
    Args:
        bbox: [west, south, east, north] in WGS-84 degrees (up to ~50 km a side at 20 m;
            larger areas are computed on a coarser grid).
        start_date: first day of the event, YYYY-MM-DD.
        end_date: last day of the event, YYYY-MM-DD.
        pad_days: days searched before and after the event for pre/post images.
        refresh: recompute even if a saved result exists for this study.

    Returns:
        {"status": "ok" | "failed", "summary", "metrics", "details", "artifacts",
         "reused": a saved result was returned, "ran": analyses run by this call,
         "study_dir": where the results and maps of this study are saved}."""


@app.tool(description=f"{ANALYSES['optical_conditions']}{_ARGS}")
async def optical_conditions(bbox: list[float], start_date: str, end_date: str,
                             pad_days: int = 14, refresh: bool = False) -> dict:
    return await _run("optical_conditions", bbox, start_date, end_date, pad_days, refresh)


@app.tool(description=f"{ANALYSES['s2_dnbr_burned_area']} Runs optical_conditions first if this "
                      f"study has no result for it.{_ARGS}")
async def s2_dnbr_burned_area(bbox: list[float], start_date: str, end_date: str,
                              pad_days: int = 14, refresh: bool = False) -> dict:
    return await _run("s2_dnbr_burned_area", bbox, start_date, end_date, pad_days, refresh)


@app.tool(description=f"{ANALYSES['s1_backscatter_change']}{_ARGS}")
async def s1_backscatter_change(bbox: list[float], start_date: str, end_date: str,
                                pad_days: int = 14, refresh: bool = False) -> dict:
    return await _run("s1_backscatter_change", bbox, start_date, end_date, pad_days, refresh)


@app.tool(description=f"{ANALYSES['s1_s2_agreement']} Reuses the burned-area maps saved by "
                      "s2_dnbr_burned_area and s1_backscatter_change for the same study (runs them "
                      "if missing); writes a quicklook PNG (artifacts). Agreement is not accuracy."
                      f"{_ARGS}")
async def s1_s2_agreement(bbox: list[float], start_date: str, end_date: str,
                          pad_days: int = 14, refresh: bool = False) -> dict:
    return await _run("s1_s2_agreement", bbox, start_date, end_date, pad_days, refresh)


@app.tool()
def list_analyses() -> dict:
    """The analyses this server runs, what each computes, and which it needs first."""
    return {name: {"description": text, "needs": list(DEPENDS.get(name, ()))}
            for name, text in ANALYSES.items()}


if __name__ == "__main__":
    app.run(transport="stdio")
