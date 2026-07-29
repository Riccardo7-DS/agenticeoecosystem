"""Single-agent baseline: one ADK Agent with every sub-agent's MCP toolset
attached directly (no routing/delegation layer), for comparison against the
multi-agent supervisor in multi_agent.py.

Structurally this mirrors multi_agent.py (same patches, same session/runner
scaffolding, same get_ans_tool_path harness hook) but replaces `sub_agents=
ALL_SUB_AGENTS` with a flat `tools=ALL_TOOLS` list pulled directly from each
sub_agent_*.py module's own `toolset`, so the exact same MCP server
processes, timeouts, and tool_filter/local-function extras are reused
verbatim — this is not a re-implementation of any server wiring, only a
different arrangement of the same building blocks.

The system prompt below is a condensed synthesis of the 10 sub-agent
instructions plus the orchestrator's routing/output/scope rules, since a
single agent (unlike the supervisor) has no sub-agent to hand domain-specific
know-how off to and must carry all of it itself.

VLM handling — deliberately NOT reused from sub_agent_vlm.py: that sub-agent
gets raw image bytes into ITS OWN context because it runs on a separate
vision-capable model (Qwen3-VL, see vlm_init.py); `llm` (EVE-Instruct) here is
not known to be multimodal, so handing it the same raw `load_images` tool
would silently misbehave. Instead this file (only this file — vlm_tools/
server.py, sub_agent_vlm.py, and vlm_init.py are untouched):
  - reuses the existing `resolve_images` MCP tool as-is, filtered via
    tool_filter so the raw, image-bytes-returning `load_images` tool from
    that same server is never exposed to this agent; and
  - defines one new self-contained tool, `describe_images`, right below,
    which resolves + loads images and asks the Qwen3-VL endpoint the
    question INTERNALLY, returning only a plain-text answer — so the
    orchestrating model never has to be vision-capable itself.
"""

from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
apply_eve_patch()

from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as apply_mcp_image_patch
apply_mcp_image_patch()

from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard as _orphan_guard
_orphan_guard.apply()

from agenticeoecosystem.agents.subagents import arxiv_rate_limit_retry as _arxiv_retry
_arxiv_retry.apply()

import asyncio
import base64
import json
import logging
import os
import uuid
import warnings

import litellm
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters
from dotenv import load_dotenv

from agenticeoecosystem.definitions import ROOT_DIR
from agenticeoecosystem.agents.subagents.llm_init import llm

from agenticeoecosystem.agents.subagents.sub_agent_cdse import toolset as _cdse_toolset
from agenticeoecosystem.agents.subagents.sub_agent_climate import toolset as _climate_toolset
from agenticeoecosystem.agents.subagents.sub_agent_eodashboard import toolset as _eodashboard_toolset
from agenticeoecosystem.agents.subagents.sub_agent_eumetsat import toolset as _eumetsat_toolset
from agenticeoecosystem.agents.subagents.sub_agent_gee import toolset as _gee_toolset
from agenticeoecosystem.agents.subagents.sub_agent_gis import (
    toolset as _gis_toolset,
    clip_raster_with_bbox,
    compute_evi,
    compute_ndwi,
    compute_ndmi,
    compute_nbr,
)
from agenticeoecosystem.agents.subagents.sub_agent_graph_rag import toolset as _graph_rag_toolset
from agenticeoecosystem.agents.subagents.sub_agent_arxiv_rag import toolset as _arxiv_toolset
from agenticeoecosystem.agents.subagents.sub_agent_mooc import toolset as _mooc_toolset

# NOT reusing sub_agent_vlm.py's toolset — that one exposes the raw
# `load_images` tool unfiltered, which is only safe for a vision-capable
# calling model. Point at the exact same, unmodified MCP server
# (vlm_tools/server.py) but filter to just `resolve_images` (text-only
# discovery); the actual vision reasoning happens in `describe_images` below.
_vlm_toolset = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/vlm_tools/server.py"],
            ),
            timeout=300,
        ),
        tool_filter=["resolve_images"],
    )
]

# Reuse vlm_tools/server.py's own (private, unmodified) discovery/GeoTIFF
# helpers so `describe_images` below loads images identically to
# `load_images` — without editing that shared file.
from agenticeoecosystem.servers.vlm_tools.server import (
    _discover as _vlm_discover,
    _geotiff_to_png_bytes,
    _GEOTIFF_EXTENSIONS,
    _MIME_TYPES,
)

warnings.filterwarnings("ignore")
load_dotenv()
logger = logging.getLogger(__name__)

# Same model/endpoint as agents/subagents/vlm_init.py's `vlm` — called
# directly via litellm here since describe_images makes one self-contained
# completion, not a stateful multi-turn agent.
_VLM_MODEL = os.getenv("VLM_MODEL_NAME", "Qwen3-VL")
_VLM_ENDPOINT = os.getenv("LLAVA_ENDPOINT", "http://10.0.0.118:18000/v1")


def _encode_image_for_vlm(path) -> tuple[str, str]:
    """Return (mime_type, base64_data) for one image file, converting
    GeoTIFFs to a viewable percentile-stretched PNG first (same conversion
    load_images uses)."""
    if path.suffix.lower() in _GEOTIFF_EXTENSIONS:
        return "image/png", base64.b64encode(_geotiff_to_png_bytes(path)).decode("utf-8")
    mime = _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return mime, base64.b64encode(path.read_bytes()).decode("utf-8")


async def describe_images(path: str, question: str, limit: int = 10) -> str:
    """Ask a vision-language model a question about image(s) at a path.

    Resolves and loads the image(s) at `path` internally and sends them,
    together with `question`, to a dedicated vision-capable model
    (Qwen3-VL) — returning only its plain-text answer. Use this for any
    question about what an image visually shows (cloud cover, qualitative
    before/after comparison, flooding/smoke by eye, general description).
    Do not use it for quantitative raster analysis (NDVI, zonal stats, band
    math) — that stays with the GIS tools.

    Args:
        path: Absolute or relative path to a single image file (JPEG/PNG/
            GeoTIFF/...), or a directory of images. Directories are capped
            at 10 images, sorted by filename.
        question: What to ask about the image(s), e.g. "Is there visible
            cloud cover in this scene?" or "What changed between these two
            images?".
        limit: Max images to load from a directory (capped at 10).

    Returns:
        The vision model's plain-text answer, or a clear "ERROR: ..."
        string if the path resolves to no image or the model call fails.
    """
    try:
        resolved = _vlm_discover(path, limit)
    except (FileNotFoundError, ValueError) as e:
        return f"ERROR: {e}"

    content: list[dict] = [{"type": "text", "text": question}]
    for f in resolved:
        try:
            mime, b64 = _encode_image_for_vlm(f)
        except Exception as e:
            logger.warning("Failed to encode image %s for VLM: %s", f, e)
            continue
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    if len(content) == 1:
        return f"ERROR: no image at '{path}' could be loaded."

    try:
        response = await litellm.acompletion(
            model=f"openai/{_VLM_MODEL}",
            api_base=_VLM_ENDPOINT,
            api_key="EMPTY",
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        logger.warning("VLM call failed for '%s': %s", path, e)
        return f"ERROR: VLM call failed: {e}"

    return response.choices[0].message.content or ""

# ---------------------------------------------------------------------------
# Flatten every sub-agent's toolset onto one tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = (
    _cdse_toolset
    + _climate_toolset
    + _eodashboard_toolset
    + _eumetsat_toolset
    + _gee_toolset
    + _gis_toolset
    + _graph_rag_toolset
    + _arxiv_toolset
    + _mooc_toolset
    + _vlm_toolset
    + [clip_raster_with_bbox, compute_evi, compute_ndwi, compute_ndmi, compute_nbr, describe_images]
)

# ---------------------------------------------------------------------------
# System prompt — condensed synthesis of all 10 sub-agent instructions plus
# the orchestrator's cross-cutting routing/output/scope rules.
# ---------------------------------------------------------------------------

_SINGLE_AGENT_SYSTEM_PROMPT = ("""You are a single Earth Observation / GIS assistant with direct access to
every tool below — there is no routing layer, so you must pick the right
tool(s) yourself and chain their outputs across domains as needed.

=== CDSE (Copernicus Data Space Ecosystem) — Sentinel-1/2/3/5P, Landsat, DEM, geocoding ===
- Place name -> bbox: geocode_place (use top_result's bbox_array). Lat/lon ->
  place name: reverse_geocode_place. For macro-regions (continents, seas,
  climate zones, e.g. "the Sahel") that geocode_place can't resolve, try
  resolve_named_region next. If both fail, infer an approximate bbox
  yourself from general geographic knowledge rather than giving up.
- Pick the fetch_* tool for the requested sensor: fetch_sentinel1_grd (SAR),
  fetch_sentinel2 (optical, specify L1C/L2A), fetch_sentinel3_olci,
  fetch_sentinel3_slstr, fetch_sentinel3_syn_l2, fetch_sentinel5p_l2 (trace
  gases), fetch_landsat_ot_l1, fetch_dem (elevation).
- Sentinel-2 has no data before June 2015 — use fetch_landsat_ot_l1 instead
  for earlier dates (it's L1/TOA only; if the user needs true L2 surface
  reflectance, use GEE instead of silently substituting TOA).
- Prefer each tool's `preset` (true_color, ndvi, elevation_color, ...) over
  custom_bands or a raw evalscript when it fits.
- as_geotiff=False (JPEG quicklook, <=3 bands) only when the output is purely
  for visual inspection. Set as_geotiff=True whenever: >3 custom_bands are
  requested; the preset is a derived index/quantity (ndvi, otci, trace-gas
  value) rather than a display composite; or the result feeds anything else
  (NDVI/band math, zonal stats, reprojection, other GIS operations).
- bbox = [west, south, east, north] in WGS-84; dates = YYYY-MM-DD.

=== Climate (ERA5 reanalysis via Copernicus CDS) ===
- Discovery: list_era5_single_level_variables, list_era5_pressure_level_variables,
  list_cds_datasets. Inspect any local .nc with inspect_netcdf.
- Monthly means: fetch_era5_single_levels / fetch_era5_pressure_levels.
  Hourly: fetch_era5_hourly_single_levels / fetch_era5_hourly_pressure_levels /
  fetch_era5_land (0.1 deg, land-only — soil/runoff/high-res precip).
  Pre-aggregated daily stats: fetch_era5_daily_statistics (prefer this over
  fetching hourly + aggregating yourself when only a daily figure is needed).
  Anything else (CMIP6, seasonal forecasts, ...): fetch_cds_dataset.
- Processing: convert_hourly_to_daily (method='sum' for precipitation,
  'mean' for temperature) -> convert_daily_to_monthly -> resample_temporal;
  convert_units (K->C, m->mm, Pa->hPa, ...); clip_to_bbox.
- Always set `area` to a bbox for a regional request. Issue large requests
  as one call first; only split into smaller sequential calls if a request
  actually times out or fails.
- Do not stop once a fetch tool returns a file — check the request for every
  implied processing step (aggregation, unit conversion, clipping, index
  computation) and complete each one before answering.

=== EO Dashboard (narratives & STAC catalog) ===
- list_narratives / list_themes / search_narratives to discover narratives;
  get_narrative_content for full text; get_narrative_indicators and
  find_narratives_for_indicator to cross-link narratives <-> STAC
  collections; search_collections / get_collection_items for STAC data.
- Always report titles, themes, tags, URLs, and indicators/collections;
  state clearly (don't silently return empty) when nothing is found.

=== EUMETSAT (MTG FCI / MSG SEVIRI geostationary products only) ===
- list_eumetsat_products to look up exact product keys if unsure.
- search_eumetsat_products BEFORE download_eumetsat_product to confirm the
  match count is reasonable (download refuses >20 matches/call — narrow the
  date range/bbox/observations_per_day instead of retrying blindly).
- Use the `path` EUMETSAT actually returns for each downloaded product,
  never a path you construct yourself (extension varies by product).
- bbox = [west, south, east, north] WGS-84 (converted internally — never
  pass EUMETSAT's native order). Dates are ISO datetimes 'YYYY-MM-DDTHH:MM:SS' UTC.
- method="datastore" (fast, raw) by default; "datatailor" only when
  reprojection/clip/format conversion is explicitly needed.
- To inspect/read values from a downloaded file, call read_eumetsat_product
  (or read=true on download). Never hand a raw datastore/datatailor file to
  a GIS tool — only pass a geotiff_path from read_eumetsat_product(as_geotiff=true).
- Not for: place geocoding, Sentinel/Landsat/DEM quicklooks (use CDSE), or
  NDVI/reprojection/zonal stats (use GIS tools on its GeoTIFF output).

=== GEE (Google Earth Engine) — for collections CDSE/EUMETSAT don't cover ===
- Use only for specialized/GEE-only collections (NASA VIIRS/MODIS, JAXA
  GPM/GSMaP/ALOS, etc.) — not for imagery CDSE or EUMETSAT already serve.
- get_lat_lon to geocode a place name first if needed.
- search_gee_catalog to find the exact, current Earth Engine dataset `id`
  for anything other than these 3 safe-from-memory IDs: COPERNICUS/S2_SR_HARMONIZED
  (Sentinel-2 SR), LANDSAT/LC08/C02/T1_L2 (Landsat 8 SR), COPERNICUS/S5P/NRTI/L3_NO2.
  GEE IDs are frequently re-versioned — don't guess a memorized ID for anything else.
- Exactly one region spec per call: latitude+longitude(+buffer_m), or
  bounding_box [min_lon,min_lat,max_lon,max_lat], or region_geojson.
  Coordinates are always [longitude, latitude] order.
- Respect each dataset's temporal_extent (from the catalog search result) —
  a date range outside it silently returns zero images.
- as_geotiff=False for an 8-bit PNG quicklook (visual inspection / VLM);
  leave as_geotiff=True (default) for downstream quantitative/GIS use — GEE's
  GeoTIFF output is not viewable by the VLM tools.

=== GIS analysis (rasterio / geopandas / shapely / pyproj / pysal) ===
- Only operate on a raster/vector file already produced earlier in THIS
  conversation (by CDSE, GEE, EUMETSAT, or ERA5/climate) — never invent or
  guess a file path.
- clip_raster_with_bbox when you only have a bounding box (no shapefile);
  clip_raster_with_shapefile only if a real .shp/.geojson path already exists.
  Before clipping to an administrative boundary, confirm that boundary file
  was actually fetched/created earlier — otherwise geocode a bbox via CDSE
  and clip to that instead.
- Spectral indices (require a multi-band raster already on disk; none of
  these fetch imagery themselves; Sentinel-2 band map: B2=Blue, B3=Green,
  B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2):
    - compute_ndvi(red, nir) — general vegetation condition.
    - compute_evi(red, nir, blue, scale_factor=0.0001 for Sentinel-2 SR DN
      0-10000, else 1.0) — dense-canopy vegetation, atmosphere-corrected.
    - compute_ndwi(green, nir) — open water (McFeeters).
    - compute_ndmi(nir, swir1) — vegetation moisture / drought stress.
    - compute_nbr(nir, swir2) — burn severity; compare pre-/post-fire as dNBR.
- Also available: reprojection, raster/zonal statistics, geodesy (UTM
  zone/CRS, geodetic distance/area), map creation (create_map / create_web_map).

=== ESA EO Science Strategy knowledge graph (EOSS GraphRAG) ===
- search_eoss for natural-language questions (returns chunks + KG context).
- query_ontology for structured, read-only Cypher (MATCH ... RETURN) lookups.
- list_ontology_nodes to enumerate nodes of a label, or all node types with
  counts if no label given. Return the raw context rather than over-summarizing.

=== ArXiv (scientific literature) ===
- search_papers to discover; prefer OR'd phrasings over strict AND'd quoted
  phrases (too strict tends to zero out real results). For EO/remote-sensing/
  climate topics, useful categories are physics.ao-ph, physics.geo-ph, and
  sometimes eess.IV/eess.SP for a processing technique — omit `categories`
  entirely if unsure rather than guessing a wrong filter (e.g. never
  eess.AS or physics.data-an for EO topics, despite appearing in examples).
- Prefer get_abstract over download_paper unless full text is explicitly
  requested. Never call download_paper for more than 1 paper, or
  get_abstract for more than 5 papers, in a single task.
- If a search returns 0 results, retry ONCE with a broadened query (AND->OR,
  or drop `categories`) before reporting no papers found.

=== EO MOOCs (educational transcripts) ===
- search_moocs to find relevant course content. Return the raw retrieved
  context — do not summarize or interpret it yourself.

=== VLM image inspection (visual reasoning over files already on disk) ===
- Use describe_images(path, question) for any question about what an image
  visually shows (cloud cover, qualitative before/after comparison,
  flooding/smoke by eye, general description) — NOT for quantitative raster
  analysis (NDVI, zonal stats, band math), which stays in the GIS tools
  above. It resolves and loads the image(s) and asks a vision model
  internally, returning only its text answer — you never see raw image
  bytes yourself.
- Optionally call resolve_images on a path first if you want to confirm
  what it resolves to (e.g. how many images are in a directory) before
  calling describe_images on it.
- If comparing multiple distinct paths, call describe_images once per path
  (not all paths at once) and compare the returned descriptions yourself.

=== CROSS-CUTTING WORKFLOW RULES ===
- Multi-clause requests ("do X, then use the result to do Y"): identify every
  clause up front and do not produce a final answer until all of them are
  addressed — satisfying only the first clause is not a complete answer.
- Do not issue two calls to the same tool in the same turn with different
  parameters (e.g. a "before" and "after" image from the same sensor/tool) —
  issue them one at a time, waiting for each result, so parameters from one
  call can't bleed into the other.
- Decide up front whether imagery is for visual inspection or downstream
  quantitative/GIS use, and set as_geotiff/format accordingly on the CDSE or
  GEE fetch call — do not fetch once and try to repurpose the wrong format.
- Prefer CDSE by default for common sensor imagery (Sentinel/Landsat/DEM).
  Use EUMETSAT only for MTG FCI/MSG SEVIRI. Use GEE only for collections
  neither of those covers. Use the Climate tools only for atmospheric/
  reanalysis variables, not imagery.
- A purely narrative/text search (EO Dashboard, ArXiv, MOOCs, EOSS) needs no
  GIS or imagery tool call at all.

=== OUTPUT RULES ===
- Return tool results directly and completely, word for word — no added
  introductions, summaries, or commentary beyond what's needed to answer.
- Report local file paths as plain text, never as markdown links or URLs.
- If a tool call returns no results or fails, say so clearly instead of
  guessing, inventing data, or silently retrying more than the specific
  retry allowances noted above (e.g. arXiv's single broadened retry).

CONVERSATION: For greetings, thanks, or small talk, respond naturally and
briefly without calling any tools.

SCOPE: Only decline requests that are both (a) unrelated to Earth
Observation, satellite data, GIS, scientific literature, or ESA/space
topics, and (b) not conversational (e.g. restaurant recommendations,
financial trading, cooking, sports scores). For those, call no tools and
respond exactly: 'This question is outside the scope of the EO/GIS assistant.'"""
)


def _on_tool_error(tool, args, tool_context, error):
    """Turn a tool-execution failure into a normal tool result instead of
    letting it propagate and abort the turn (mirrors multi_agent.py)."""
    logger.warning("Tool '%s' failed: %s", getattr(tool, "name", "?"), error)
    return {"error": f"Tool '{getattr(tool, 'name', '?')}' failed: {error}"}


root = Agent(
    model=llm,
    name="single_agent",
    description=(
        "A single agent with direct access to every EO/GIS/climate/literature "
        "tool — CDSE, GEE, EUMETSAT, ERA5/Climate, GIS analysis, EO Dashboard, "
        "EOSS GraphRAG, ArXiv, MOOCs, and VLM image inspection."
    ),
    instruction=_SINGLE_AGENT_SYSTEM_PROMPT,
    tools=ALL_TOOLS,
    on_tool_error_callback=_on_tool_error,
)

APP_NAME = "EVE-AGENT-ECOSYSTEM-SINGLE"
USER_ID = "user_1"
SESSION_ID = "session_001"

session_service = InMemorySessionService()


async def init_session(app_name: str, user_id: str, session_id: str) -> InMemorySessionService:
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    logger.info("Session created: App='%s', User='%s', Session='%s'", app_name, user_id, session_id)
    return session


session = asyncio.run(init_session(APP_NAME, USER_ID, SESSION_ID))

runner = Runner(agent=root, app_name=APP_NAME, session_service=session_service)
logger.info("Runner created for agent '%s'.", runner.agent.name)


def _extract_final_text(event) -> str | None:
    if not (event.content and event.content.parts):
        return None
    for part in event.content.parts:
        text = getattr(part, "text", None)
        if text:
            return text
    return None


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    logger.info(f"\n>>> User Query: {query}")
    print(f"\n>>> User Query: {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.author:
            print(f"\n[AGENT] {event.author}")

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    print(f"  [TOOL CALL] {fc.name}({dict(fc.args)})")
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response_str = str(fr.response)
                    if len(response_str) > 300:
                        response_str = response_str[:300] + "... [truncated]"
                    print(f"  [TOOL RESULT] {fr.name} -> {response_str}")

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_response_text = text
            elif event.actions and event.actions.escalate:
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    print(f"\n<<< Agent Response: {final_response_text}")


async def get_ans_tool_path(query: str) -> str:
    """
    Runs a query through the single agent in an isolated session and returns
    a JSON string with the final answer, the leaf MCP tool calls (name +
    arguments) made to produce it, and total token usage across the run.
    Same shape as multi_agent.get_ans_tool_path, minus `agents_called` (there
    is only ever one agent here).
    """
    _orphan_guard.reset()

    session_id = f"session_{uuid.uuid4().hex}"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_answer = "Agent did not produce a final response."
    tools_called = []
    input_tokens = 0
    output_tokens = 0

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.usage_metadata:
            input_tokens += event.usage_metadata.prompt_token_count or 0
            output_tokens += event.usage_metadata.candidates_token_count or 0

        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tools_called.append(
                        {"function_name": fc.name, "arguments": dict(fc.args)}
                    )

                fr = getattr(part, "function_response", None)
                if fr is not None:
                    resp = fr.response
                    error_text = None
                    if isinstance(resp, dict):
                        if resp.get("error"):
                            error_text = str(resp["error"])
                        elif resp.get("isError"):
                            error_text = str(resp.get("content") or resp)
                    for entry in reversed(tools_called):
                        if entry["function_name"] == fr.name and "error" not in entry:
                            if error_text:
                                entry["error"] = error_text
                            break

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_answer = text
            elif event.actions and event.actions.escalate:
                final_answer = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    ended_in_error = bool(tools_called) and "error" in tools_called[-1]

    return json.dumps({
        "final_answer": final_answer,
        "ended_in_error": ended_in_error,
        "tools_called": tools_called,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }, default=str)


async def run_conversation():
    await call_agent_async(
        "Search arXiv for papers on sea-level rise monitoring using satellite altimetry published since 2024 and return up to 15 results.",
        runner, USER_ID, SESSION_ID,
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
