"""Per-sub-agent knowledge used by the reviewer and the hypothesis agent.

RULES: deterministic checks on a recorded interaction (free; run on every review).
AGENT_CAPABILITIES: what each sub-agent can do. The checklist uses it so it doesn't demand
impossible things; the hypothesis agent uses it to address its NEXT QUESTIONS.
Avoid {braces} in descriptions: ADK reads them as state variables in instructions.
"""
import json
import re
from typing import Callable

from agenticeoecosystem.agents.helper.trace import Interaction

# ------------------------------- climate_agent -------------------------------

def _climate_area_rule(it: Interaction) -> list[str]:
    return [f"step {s.index}: {s.tool} without `area` (global download). OK only if global was requested."
            for s in it.steps
            if s.tool.startswith(("fetch_era5_hourly", "fetch_era5_land", "fetch_era5_daily"))
            and not s.args.get("area")]


def _climate_precip_sum_rule(it: Interaction) -> list[str]:
    return [f"step {s.index}: precipitation aggregated with method={s.args.get('method')!r}, expected 'sum'."
            for s in it.steps
            if s.tool == "convert_hourly_to_daily" and "precip" in json.dumps(s.args).lower()
            and s.args.get("method", "sum") != "sum"]


def _files_exist_rule(it: Interaction) -> list[str]:
    produced = " ".join(json.dumps(s.args) + s.response for s in it.steps)
    return [f"answer mentions {p} but no tool step produced it"
            for p in set(re.findall(r"[\w./-]+\.nc\b", it.answer or "")) if p not in produced]

# ------------------------------ arxiv_rag_agent ------------------------------
# ASSUMPTION: search_papers output contains "total_results": <n>. Check one real response.
_ZERO_RESULTS = re.compile(r"total_results\W{0,3}\s*0\b")


def _arxiv_download_cap(it: Interaction) -> list[str]:
    n = sum(s.tool == "download_paper" for s in it.steps)
    return [f"download_paper called {n} times; the prompt allows at most 1"] if n > 1 else []


def _arxiv_abstract_cap(it: Interaction) -> list[str]:
    n = sum(s.tool == "get_abstract" for s in it.steps)
    return [f"get_abstract called {n} times; the prompt allows at most 5"] if n > 5 else []


def _arxiv_zero_retry(it: Interaction) -> list[str]:
    searches = [s for s in it.steps if s.tool == "search_papers"]
    if len(searches) == 1 and _ZERO_RESULTS.search(searches[0].response):
        return ["search returned 0 results and was not retried with a broadened query"]
    return []


# DECISION: one list per sub-agent.
RULES: dict[str, list[Callable[[Interaction], list[str]]]] = {
    "climate_agent": [_climate_area_rule, _climate_precip_sum_rule, _files_exist_rule],
    "arxiv_rag_agent": [_arxiv_download_cap, _arxiv_abstract_cap, _arxiv_zero_retry],
}

AGENT_CAPABILITIES: dict[str, str] = {
    "cdse_agent": "Satellite quicklooks and GeoTIFFs from the Copernicus Data Space Ecosystem "
                  "(Sentinel-1/2/3, Sentinel-5P, Landsat, DEM); place name to bbox geocoding.",
    "eumetsat_agent": "Search and download of geostationary EUMETSAT products (MTG FCI and MSG SEVIRI "
                      "only); per-band statistics of downloaded files.",
    "gee_agent": "Google Earth Engine downloads for collections CDSE and EUMETSAT do not cover "
                 "(e.g. VIIRS, MODIS, GPM/GSMaP, ALOS).",
    "climate_agent": "Fetches and processes ERA5/CDS reanalysis data (download, temporal aggregation, "
                     "unit conversion, clipping, regional time-series summaries). Returns file paths and "
                     "NetCDF summaries; the summaries are relayed BY DESIGN and count as on topic.",
    "eo_stac_agent": "EO Dashboard (ESA/NASA/JAXA) narratives, themes and STAC collections.",
    "graph_rag_agent": "Retrieval from the ESA EO Science Strategy: science themes and questions, "
                       "strategic objectives, tipping points, treaties.",
    "mooc_rag_agent": "Semantic search over Earth Observation online course transcripts.",
    "arxiv_rag_agent": "ArXiv search, abstracts (at most 5 per task) and full text (at most 1 per task).",
    "gis_agent": "Spatial and raster analysis of files already on disk (clipping, reprojection, zonal "
                 "statistics, spectral indices). Cannot download data.",
    "vlm_agent": "Visual inspection of image files already on disk, up to 10 at a time.",
    "tle_agent": "Satellite positions and upcoming overpasses of a location, from TLEs.",
}