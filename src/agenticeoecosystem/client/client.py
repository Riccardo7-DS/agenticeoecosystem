import asyncio
import logging
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from agenticeoecosystem.config.mcp_config import get_mcp_servers
from agenticeoecosystem.config.credentials import get_langchain_chat_openai

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step log — populated during a pipeline run
# ---------------------------------------------------------------------------

_step_log: list[dict] = []
_token_counts: dict[str, int] = {"input": 0, "output": 0}


def reset_step_log() -> None:
    _step_log.clear()
    _token_counts["input"] = 0
    _token_counts["output"] = 0


def _extract_steps_from_messages(messages: list) -> None:
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            _token_counts["input"] += msg.usage_metadata.get("input_tokens", 0)
            _token_counts["output"] += msg.usage_metadata.get("output_tokens", 0)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                _step_log.append({"function_name": tc["name"], "arguments": tc["args"]})


async def _load_tools_from_mcp():
    logger.info("Building MCP server configuration")
    mcp_servers = get_mcp_servers()
    logger.info("MCP servers configured: %s", list(mcp_servers.keys()))

    all_tools: list = []
    for server_name, server_config in mcp_servers.items():
        try:
            tools = await MultiServerMCPClient({server_name: server_config}).get_tools()
            all_tools.extend(tools)
            logger.info("  [%s] %d tool(s) loaded", server_name, len(tools))
        except Exception as exc:
            logger.warning("  [%s] skipped — failed to load tools: %s", server_name, exc)

    logger.info("MCP tool discovery complete: %d tools", len(all_tools))
    for t in all_tools:
        logger.debug("  - %s: %s", t.name, t.description)
    return all_tools


import nest_asyncio
nest_asyncio.apply()
mcp_tools = asyncio.get_event_loop().run_until_complete(_load_tools_from_mcp())

# ---------------------------------------------------------------------------
# Tool name sets
# ---------------------------------------------------------------------------

_EO_STAC_TOOL_NAMES = {
    "list_narratives", "search_narratives", "get_narrative_content",
    "list_themes", "get_narrative_indicators", "search_collections",
    "get_collection_items", "find_narratives_for_indicator",
}

_ARXIV_TOOL_NAMES = {
    "search_papers", "download_paper", "list_papers", "read_paper",
    "get_abstract", "semantic_search", "reindex", "citation_graph",
    "watch_topic", "check_alerts",
}

_GIS_TOOL_NAMES = {
    # Shapely geometry
    "buffer", "intersection", "union", "difference", "symmetric_difference",
    "convex_hull", "envelope", "minimum_rotated_rectangle", "centroid",
    "bounds", "coordinates", "geometry_type", "rotate", "scale", "translate",
    "triangulate", "voronoi", "unary_union", "length", "area",
    "is_valid", "make_valid", "simplify", "snap", "nearest_point",
    "normalize", "geometry_to_geojson", "geojson_to_geometry",
    # PyProj coordinate transforms
    "transform_coordinates", "project_geometry",
    "get_crs_info", "get_available_crs", "get_utm_zone", "get_utm_crs", "get_geocentric_crs",
    "get_geod_info", "calculate_geodetic_distance", "calculate_geodetic_point", "calculate_geodetic_area",
    # GeoPandas vector
    "read_file_gpd", "write_file_gpd", "append_gpd", "merge_gpd",
    "overlay_gpd", "dissolve_gpd", "explode_gpd", "clip_vector",
    "sjoin_gpd", "sjoin_nearest_gpd", "point_in_polygon",
    # Rasterio raster
    "metadata_raster", "get_raster_crs", "extract_band", "raster_band_statistics",
    "raster_histogram", "clip_raster_with_shapefile", "resample_raster",
    "reproject_raster", "tile_raster", "compute_ndvi", "raster_algebra",
    "concat_bands", "weighted_band_sum", "zonal_statistics",
    "reclassify_raster", "focal_statistics", "hillshade", "write_raster",
    # PySAL spatial statistics
    "morans_i", "gearys_c", "gamma_statistic", "getis_ord_g",
    "moran_local", "getis_ord_g_local", "join_counts_local", "adbscan",
    "weights_from_shapefile", "distance_band_weights", "knn_weights",
    "build_transform_and_save_weights", "ols_with_spatial_diagnostics_safe",
    "build_and_transform_weights", "spatial_markov", "dynamic_lisa", "gm_lag",
    # Visualization
    "create_map", "create_web_map",
    # Data downloads
    "download_boundaries", "download_climate_data",
    "get_species_info", "download_species_occurrences",
    "download_street_network", "calculate_shortest_path",
    "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_EO_STAC_SYSTEM_PROMPT = (
    "You are a EO Dashboard assistant. "
    "Parse natural language queries (e.g., 'Find narrative about Covid-19') "
    "and look for values that can be used as inputs to the tools you've at your disposal. "
    """
    EO Dashboard MCP Tools Usage Instructions:

    1. Listing and Exploring Narratives:
    - Use list_narratives to retrieve all available narratives.
    - Use list_themes to see what themes exist before filtering or searching.
    - Use search_narratives with a keyword or theme to find specific narratives.
    - Always confirm how many narratives were found and include their titles, themes, and URLs.

    2. Accessing Narrative Content:
    - Use get_narrative_content with the narrative URL to fetch full content.
    - Limit your processing to the first few thousand characters if the content is very long.
    - Include both frontmatter and body content when summarizing or analyzing narratives.
    - Parse information related to the data used and also include it in the output.

    3. Working with Narrative Indicators:
    - Use get_narrative_indicators to see which STAC collections (datasets) a narrative references.
    - If no indicators are found, clearly state that.
    - Use find_narratives_for_indicator to reverse lookup narratives for a specific dataset.
    - Always confirm the narrative titles, themes, and URLs when returning results.

    4. Searching and Accessing STAC Collections:
    - Use search_collections to find collections by code, theme, or keyword.
    - Use get_collection_items to retrieve sample items (observations) from a collection.
    - Always check that the collection exists before retrieving items.
    - Limit the number of items returned to avoid overwhelming output.
    - If a collection has no geoparquet asset or items cannot be retrieved, report clearly.

    5. General Workflow Guidance:
    - Always load and cache narratives/catalog first with EODashboardCache methods.
    - Make sure to chain the tools and remember to parse the output from one tool to get input to another tool.
    - Present all results clearly: titles, themes, tags, dates, URLs, and indicators/collections where applicable.
    - If a search returns no results, explicitly confirm unavailability instead of returning empty data.
    """
)

_GEE_SYSTEM_PROMPT = (
    "You are an expert GEE assistant responsible for translating user requests into precise tool calls.\n"
    "Your task is to answer the user's query about satellite image downloads by using the `download_satellite_image` tool.\n"
    "Additionally, you can use the `get_lat_lon` tool to obtain the latitude and longitude of a given address.\n"
    "REQUIRED FORMAT RULES:\n"
    "- start_date: string (YYYY-MM-DD)\n"
    "- end_date: string (YYYY-MM-DD)\n"
    "- bounding_box: [min_lon, min_lat, max_lon, max_lat]\n"
    "- bands: MUST be a comma-separated string (e.g., \"B4,B3,B2,B8,B11\") — NOT a list\n"
    "- scale: integer\n"
    "- image_count: integer\n"
    "- max_cloud_cover: integer\n"
    "Extract all other parameters like dates, cloud cover, and image count from the query.\n"
    "CRITICAL: After the tool returns, your final response MUST include the exact absolute file path(s) "
    "from the 'files' list in the tool result. Format them clearly, e.g.: "
    "'Downloaded files: [\"/path/to/gee_image_20230615_t1_n1.tif\"]'. "
    "Never omit the file paths — they are required for downstream GIS processing."
)

_EVE_RETRIEVAL_SYSTEM_PROMPT = (
    "You are a knowledge retrieval assistant for Earth Observation and ESA topics. "
    "Use the retrieve tool to find relevant information and answer the user's question."
)

_RAG_SYSTEM_PROMPT = (
    "You are a retrieval assistant for ESA Earth Observation Science Strategy. "
    "Use the hybrid_rag_retrieve tool to fetch relevant document chunks and "
    "knowledge graph context, then return the raw context to the caller."
)

_ARXIV_SYSTEM_PROMPT = (
    "You are an assistant for querying and retrieving scientific papers from an ArXiv MCP server. "
    "Use the available tools to search for papers, retrieve abstracts, download full texts, and perform semantic searches. "
    "When a user asks a question, determine which tool(s) to use and extract the necessary parameters from the query. "
    "For example, if a user asks 'Find recent papers on climate change and get their abstracts', you might use "
    "'search_papers' with the keyword 'climate change' and a recent date filter, and proper category, "
    "then use 'get_abstract' for each relevant paper found.\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Prefer 'get_abstract' over 'download_paper' unless the user explicitly requests the full text. "
    "  Abstracts contain enough information for most questions and avoid context overflows.\n"
    "- NEVER call 'download_paper' for more than one paper in a single task — full PDFs are very large "
    "  and will exceed the model's context window.\n"
    "- If asked to summarise or analyse multiple papers, use 'get_abstract' for all of them."
)

_GIS_SYSTEM_PROMPT = (
    "You are a GIS analysis assistant. Use the available tools to perform spatial operations, "
    "raster analysis, coordinate transformations, spatial statistics, map visualization, and geospatial data downloads. "
    "Available capabilities include:\n"
    "- Geometry operations (buffer, intersect, union, centroid, simplify, etc.) via Shapely\n"
    "- Vector file operations (read/write/clip/merge shapefiles and GeoJSON) via GeoPandas\n"
    "- Raster analysis (NDVI, zonal statistics, clip, reproject, hillshade) via Rasterio\n"
    "- Coordinate transforms and geodetic calculations via PyProj\n"
    "- Spatial statistics (Moran's I, LISA, spatial regression) via PySAL\n"
    "- Map creation (static PNG/PDF via create_map, interactive HTML via create_web_map)\n"
    "- Data downloads (GADM boundaries, ESA WorldCover, ERA5 climate, OSM street networks, "
    "Sentinel-2 NDVI via Planetary Computer)\n"
    "Extract all parameters from the user's request and chain tools as needed. "
    "Return file paths for saved outputs (maps, rasters, vectors)."
)

_SUPERVISOR_SYSTEM_PROMPT = (
    "You are a routing agent. "
    "You can get data from EO Dashboards, Google Earth Engine, "
    "the EVE knowledge base for ESA/EO background information, "
    "the ESA EO Science Strategy knowledge base (hybrid RAG), "
    "the ArXiv MCP server for scientific papers, "
    "and the GIS analysis server for spatial operations, raster analysis, visualization, and geospatial data downloads. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence. "
    "IMPORTANT: Return the tool result directly and completely to the user, word for word. "
    "Do not add introductions, summaries, or commentary around tool results.\n"
    "SCOPE RULE: If a request is clearly unrelated to Earth Observation, satellite data, GIS analysis, "
    "scientific literature, or ESA/space topics (e.g. restaurant recommendations, financial trading, "
    "general cooking, sports scores), do NOT call any tools. "
    "Instead respond: 'This question is outside the scope of the EO/GIS assistant.'"
)

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def load_llm(model_name: str, provider: str = "eve", temperature: float = 0.0):
    """Build a LangChain chat model.

    provider options:
      "eve"    — EVE OpenAI-compatible endpoint with Basic Auth (default)
      "openai" — standard OpenAI API (requires OPENAI_API_KEY in .env)
      "custom" — any OpenAI-compatible endpoint (requires OPENAI_ENDPOINT + OPENAI_API_KEY)
      "vllm"   — local vLLM server (requires VLLM_HOST / VLLM_PORT in .env, defaults 127.0.0.1:8000)
      "ollama" — local Ollama instance
      any other string is passed directly to LangChain's init_chat_model as model_provider
    """
    if provider in ("eve", "openai", "custom"):
        return get_langchain_chat_openai(model_name, temperature=temperature, provider=provider)
    if provider == "vllm":
        from agenticeoecosystem.utils.registry import get_vllm_langchain_model, MODEL_REGISTRY
        host = os.getenv("VLLM_HOST", "127.0.0.1")
        port = int(os.getenv("VLLM_PORT", "8000"))
        key = model_name if model_name in MODEL_REGISTRY else None
        if key:
            return get_vllm_langchain_model(key, host=host, port=port, temperature=temperature)
        return ChatOpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="EMPTY",
            model=model_name,
            temperature=temperature,
        )
    return init_chat_model(model_name, model_provider=provider, temperature=temperature)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

_DEFAULT_MODEL: str = os.getenv("MAIN_MODEL_NAME", "EVE-Instruct")
_DEFAULT_PROVIDER: str = os.getenv("MODEL_PROVIDER", "eve")

model = load_llm(_DEFAULT_MODEL, _DEFAULT_PROVIDER)

eo_stac_agent       = create_agent(model, tools=[t for t in mcp_tools if t.name in _EO_STAC_TOOL_NAMES], system_prompt=_EO_STAC_SYSTEM_PROMPT)
gee_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in ["download_satellite_image", "get_lat_lon"]], system_prompt=_GEE_SYSTEM_PROMPT)
eve_retrieval_agent = create_agent(model, tools=[t for t in mcp_tools if t.name == "retrieve"], system_prompt=_EVE_RETRIEVAL_SYSTEM_PROMPT)
eoss_rag_agent      = create_agent(model, tools=[t for t in mcp_tools if t.name == "hybrid_rag_retrieve"], system_prompt=_RAG_SYSTEM_PROMPT)
arxiv_agent         = create_agent(model, tools=[t for t in mcp_tools if t.name in _ARXIV_TOOL_NAMES], system_prompt=_ARXIV_SYSTEM_PROMPT)
gis_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in _GIS_TOOL_NAMES], system_prompt=_GIS_SYSTEM_PROMPT)


@tool
async def get_eo_stac_data(request: str) -> str:
    """Get data from EO Dashboard and STAC catalogs.

    Handles cross-platform queries: e.g. fetch STAC collection data for a narrative
    by searching narratives, picking one, parsing the collection code, then getting items.
    """
    logger.info("[TOOL CALL] get_eo_stac_data: %s", request)
    try:
        result = await eo_stac_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_eo_stac_data: %s", e)
        return f"Error: {e}"


@tool
async def get_arxiv_data(request: str) -> str:
    """Get scientific papers and abstracts from ArXiv."""
    logger.info("[TOOL CALL] get_arxiv_data: %s", request)
    try:
        result = await arxiv_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_arxiv_data: %s", e)
        return f"Error: {e}"


@tool
async def get_gee_data(request: str) -> str:
    """Download satellite imagery from Google Earth Engine.

    Phrase your request like:
    "Download a Sentinel-2 image for bbox [min_lon, min_lat, max_lon, max_lat],
    date range YYYY-MM-DD to YYYY-MM-DD, bands B4,B3,B2, scale 10m, max 20% cloud cover."

    The response will include the absolute path(s) of the downloaded GeoTIFF file(s).
    Pass those paths verbatim to get_gis_analysis when further raster processing is needed.
    """
    logger.info("[TOOL CALL] get_gee_data: %s", request)
    try:
        result = await gee_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_gee_data: %s", e)
        return f"Error: {e}"


@tool
async def get_eve_knowledge(request: str) -> str:
    """Retrieve knowledge about ESA and Earth Observation from the EVE knowledge base."""
    logger.info("[TOOL CALL] get_eve_knowledge: %s", request)
    try:
        result = await eve_retrieval_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_eve_knowledge: %s", e)
        return f"Error: {e}"


@tool
async def get_hybrid_rag_context(request: str) -> str:
    """Retrieve relevant context from the ESA EO Science Strategy knowledge base.

    Uses hybrid retrieval (semantic vector search + ESA knowledge graph) over PDF documents.
    Use for questions about science themes, strategic objectives, enabling tools, and other ESA EO topics.
    """
    logger.info("[TOOL CALL] get_hybrid_rag_context: %s", request)
    try:
        result = await eoss_rag_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        for msg in result["messages"]:
            for tc in getattr(msg, "tool_calls", []):
                logger.debug("[eoss_rag_agent -> hybrid_rag_retrieve] args: %s", tc.get("args", {}))
            if getattr(msg, "type", None) == "tool":
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content
                    )
                logger.debug("[hybrid_rag_retrieve raw output]\n%s", content)
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_hybrid_rag_context: %s", e)
        return f"Error: {e}"


@tool
async def get_gis_analysis(request: str) -> str:
    """Perform GIS analysis: spatial operations, raster processing, coordinate transforms,
    spatial statistics, map visualization, and geospatial data downloads.

    Examples:
    - "Compute NDVI from a GeoTIFF raster at /path/to/image.tif"
    - "Buffer the polygon in /path/to/area.geojson by 5000 metres"
    - "Download ESA WorldCover land cover for bbox [12.4, 41.8, 12.6, 42.0]"
    - "Create a static map from /path/to/boundaries.shp and save as map.png"
    - "Calculate Moran's I for the column 'income' in /path/to/data.shp"
    """
    logger.info("[TOOL CALL] get_gis_analysis: %s", request)
    try:
        result = await gis_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        _extract_steps_from_messages(result["messages"])
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)
    except Exception as e:
        logger.error("[TOOL ERROR] get_gis_analysis: %s", e)
        return f"Error: {e}"


supervisor_agent = create_agent(
    model,
    tools=[get_eo_stac_data, get_gee_data, get_eve_knowledge, get_hybrid_rag_context, get_arxiv_data, get_gis_analysis],
    system_prompt=_SUPERVISOR_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Hot-swap agents without restarting (used by evaluation pipeline)
# ---------------------------------------------------------------------------

def rebuild_agents(model_name: str, provider: str = "ollama", temperature: float = 0.0) -> None:
    """Recreate all agents with a new model. MCP tools are reused."""
    global model, eo_stac_agent, gee_agent, eve_retrieval_agent, eoss_rag_agent, arxiv_agent, gis_agent, supervisor_agent
    model               = load_llm(model_name, provider, temperature)
    eo_stac_agent       = create_agent(model, tools=[t for t in mcp_tools if t.name in _EO_STAC_TOOL_NAMES], system_prompt=_EO_STAC_SYSTEM_PROMPT)
    gee_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in ["download_satellite_image", "get_lat_lon"]], system_prompt=_GEE_SYSTEM_PROMPT)
    eve_retrieval_agent = create_agent(model, tools=[t for t in mcp_tools if t.name == "retrieve"], system_prompt=_EVE_RETRIEVAL_SYSTEM_PROMPT)
    eoss_rag_agent      = create_agent(model, tools=[t for t in mcp_tools if t.name == "hybrid_rag_retrieve"], system_prompt=_RAG_SYSTEM_PROMPT)
    arxiv_agent         = create_agent(model, tools=[t for t in mcp_tools if t.name in _ARXIV_TOOL_NAMES], system_prompt=_ARXIV_SYSTEM_PROMPT)
    gis_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in _GIS_TOOL_NAMES], system_prompt=_GIS_SYSTEM_PROMPT)
    supervisor_agent    = create_agent(model, tools=[get_eo_stac_data, get_gee_data, get_eve_knowledge, get_hybrid_rag_context, get_arxiv_data, get_gis_analysis], system_prompt=_SUPERVISOR_SYSTEM_PROMPT)
    logger.info("Agents rebuilt — model=%s  provider=%s  temperature=%s", model_name, provider, temperature)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from agenticeoecosystem.utils.logging import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="AgenticEO client")
    parser.add_argument("--model",       default=_DEFAULT_MODEL,    help="Model name")
    parser.add_argument("--provider",    default=_DEFAULT_PROVIDER, help="Model provider")
    parser.add_argument("--temperature", type=float, default=0.0,   help="Sampling temperature")
    args = parser.parse_args()

    if args.model != _DEFAULT_MODEL or args.provider != _DEFAULT_PROVIDER or args.temperature != 0.0:
        rebuild_agents(args.model, args.provider, args.temperature)

    logger.info("All agents ready")

    async def run():
        logger.info("Entering interactive loop")
        conversation_history = []
        while True:
            user_request = input("\nQuery (or 'quit' to exit): ").strip()
            if not user_request:
                continue
            if user_request.lower() in ("quit", "exit", "q"):
                break
            logger.info("User query: %s", user_request)
            conversation_history.append({"role": "user", "content": user_request})
            print("=" * 80)
            final_ai_content = None
            async for step in supervisor_agent.astream(
                {"messages": conversation_history}  # ← pass full history
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        if message.type == "tool":
                            print(message.content)
                            logger.debug("Tool result: %s", message.content)
                        elif message.type == "ai" and message.content:
                            message.pretty_print()
                            logger.info("Final response: %s", message.content)
                            final_ai_content = message.content
            if final_ai_content:
                conversation_history.append({"role": "assistant", "content": final_ai_content})
    asyncio.run(run())
