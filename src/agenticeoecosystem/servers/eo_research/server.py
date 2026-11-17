"""
MCP server exposing the Virtual Lab's research helpers that need no credentials:

  search_catalogues                federated data discovery (agents/virtual_lab/discovery.py):
                                   collections from the Copernicus Data Space (STAC), EUMETSAT
                                   Data Store, NASA Earthdata (CMR) and Copernicus Climate Data
                                   Store, ranked by meaning against a description of the data
                                   needed, then checked for data in the bbox and period
  rank_papers_by_citations         citation counts for arXiv ids, most cited first
                                   (Semantic Scholar, falling back to OpenAlex)
  geostationary_viewing_geometry   viewing zenith angle of each geostationary imager over a bbox,
                                   with a usable / degraded / unusable verdict

The lab calls the same functions in-process. Ranking uses the OpenAI-compatible embeddings
endpoint in EMBEDDING_ENDPOINT (default http://127.0.0.1:8090/v1, scripts/serve_embeddings.py)
and falls back to word overlap when it is unreachable; the reply says which was used.
Bounding boxes are [west, south, east, north] in WGS-84 degrees.

Run with:
    python server.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from agenticeoecosystem.agents.virtual_lab.discovery import (
    PROVIDER_LABELS,
    as_tool_result,
    direct_call,
    discover_for,
)
from agenticeoecosystem.agents.virtual_lab.inventory import IODC_LON, Catalogs
from agenticeoecosystem.agents.virtual_lab.literature import rank_papers_by_citations as _rank_papers
from agenticeoecosystem.agents.virtual_lab.sensor_catalog import (
    SENSORS,
    bbox_centre,
    check_bbox,
    geometry_verdict,
    geostationary_vza_deg,
)
from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv(dotenv_path=ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, stream=sys.stderr,   # stdout is the MCP stream
                    format="%(asctime)s - EO-RESEARCH-MCP - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastMCP("eo-research-mcp")
_catalogs: Catalogs | None = None


@app.tool()
async def search_catalogues(query: str, bbox: list[float] | None = None,
                            start_date: str | None = None, end_date: str | None = None,
                            providers: list[str] | None = None) -> dict:
    """Search every data catalogue at once (Copernicus Data Space, EUMETSAT, NASA Earthdata,
    Copernicus Climate Data Store) for collections matching `query`, ranked by meaning, and
    check which have data in bbox [west, south, east, north] between start_date and end_date
    (YYYY-MM-DD).

    Write `query` the way a catalogue describes data: instrument or technique, product form,
    quantity (e.g. 'thermal infrared active fire detections with fire radiative power', or
    'gridded satellite precipitation estimates, daily'), not the question being studied.
    Without bbox and dates the collections are ranked but not checked for data.

    Args:
        query: description of the data needed.
        bbox: optional [west, south, east, north] in WGS-84 degrees.
        start_date: optional first day, YYYY-MM-DD.
        end_date: optional last day, YYYY-MM-DD.
        providers: optional subset of cdse, eumetsat, earthdata, cds.

    Returns:
        {"ranking": "embedding" | "lexical", "listings": per-provider notes,
         "collections": [{"provider", "id", "title", "match", "status", "availability"}]};
        status is available / partial / absent / unknown (a failed lookup is unknown, not absent).
    """
    global _catalogs
    if not query.strip():
        return {"error": "query is empty."}
    if bbox is not None and (error := check_bbox(bbox)):
        return {"error": error}
    unknown = [p for p in providers or [] if p not in PROVIDER_LABELS]
    if unknown:
        return {"error": f"Unknown providers {unknown}; use {', '.join(PROVIDER_LABELS)}."}
    _catalogs = _catalogs or Catalogs.public()
    found = await discover_for([query], _catalogs, bbox, start_date, end_date, direct_call,
                               providers=providers or None)
    return as_tool_result(found)


@app.tool()
async def rank_papers_by_citations(arxiv_ids: list[str]) -> dict:
    """Rank arXiv papers by citation count, most cited first (Semantic Scholar, falling back to
    OpenAlex, which can undercount preprints cited through their journal version).

    Args:
        arxiv_ids: arXiv identifiers, e.g. ["2106.09685", "1706.03762v5"].

    Returns:
        {"ranked": [{"arxiv_id", "title", "year", "citation_count", "influential_citation_count",
                     "citations_per_year"}], "source": "semantic_scholar" | "openalex"};
        papers without data come last with citation_count null; "error" if every source failed.
    """
    return await _rank_papers(arxiv_ids)


@app.tool()
def geostationary_viewing_geometry(bbox: list[float]) -> dict:
    """Viewing zenith angle of every geostationary imager (MTG FCI, MSG SEVIRI at 0 deg and over
    the Indian Ocean, GOES-East ABI) at the centre of bbox, best placed first. Above 70 deg
    pixels and slant paths degrade retrievals; above 80 deg the area is at the disk edge and not
    usable for quantitative work.

    Args:
        bbox: [west, south, east, north] in WGS-84 degrees.

    Returns:
        {"centre": [lat, lon], "imagers": [{"sensor", "sub_satellite_lon", "viewing_zenith_deg",
         "verdict", "fetch_via"}]}.
    """
    if error := check_bbox(bbox):
        return {"error": error}
    lat, lon = bbox_centre(bbox)
    geo = [(s.key, s.sub_satellite_lon, s.sub_agents) for s in SENSORS
           if s.orbit == "geostationary" and s.sub_satellite_lon is not None]
    geo += [(f"{s.key} (Indian Ocean)", IODC_LON, s.sub_agents) for s in SENSORS
            if s.key == "MSG SEVIRI"]
    rows: list[dict[str, Any]] = []
    for key, sub_lon, agents in geo:
        vza = geostationary_vza_deg(lat, lon, sub_lon)
        rows.append({"sensor": key, "sub_satellite_lon": sub_lon,
                     "viewing_zenith_deg": round(vza, 1), "verdict": geometry_verdict(vza),
                     "fetch_via": list(agents) or "no sub-agent in this system"})
    rows.sort(key=lambda r: r["viewing_zenith_deg"])
    return {"centre": [round(lat, 4), round(lon, 4)], "imagers": rows}


if __name__ == "__main__":
    app.run(transport="stdio")
