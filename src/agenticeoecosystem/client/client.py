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

# eopssrag branch — only GraphRAG tools active.
# All other tool name sets commented out; restore on main branch.

_GRAPHRAG_TOOL_NAMES = {"search_eopss", "query_ontology", "list_ontology_nodes"}

# _EO_STAC_TOOL_NAMES = {
#     "list_narratives", "search_narratives", "get_narrative_content",
#     "list_themes", "get_narrative_indicators", "search_collections",
#     "get_collection_items", "find_narratives_for_indicator",
# }
# _ARXIV_TOOL_NAMES = {
#     "search_papers", "download_paper", "list_papers", "read_paper",
#     "get_abstract", "semantic_search", "reindex", "citation_graph",
#     "watch_topic", "check_alerts",
# }
# _MOOC_TOOL_NAMES = {"search_moocs"}
# _GIS_TOOL_NAMES = {
#     "buffer", "intersection", "union", "difference", "symmetric_difference",
#     "convex_hull", "envelope", "minimum_rotated_rectangle", "centroid",
#     "bounds", "coordinates", "geometry_type", "rotate", "scale", "translate",
#     "triangulate", "voronoi", "unary_union", "length", "area",
#     "is_valid", "make_valid", "simplify", "snap", "nearest_point",
#     "normalize", "geometry_to_geojson", "geojson_to_geometry",
#     "transform_coordinates", "project_geometry",
#     "get_crs_info", "get_available_crs", "get_utm_zone", "get_utm_crs", "get_geocentric_crs",
#     "get_geod_info", "calculate_geodetic_distance", "calculate_geodetic_point", "calculate_geodetic_area",
#     "read_file_gpd", "write_file_gpd", "append_gpd", "merge_gpd",
#     "overlay_gpd", "dissolve_gpd", "explode_gpd", "clip_vector",
#     "sjoin_gpd", "sjoin_nearest_gpd", "point_in_polygon",
#     "metadata_raster", "get_raster_crs", "extract_band", "raster_band_statistics",
#     "raster_histogram", "clip_raster_with_shapefile", "resample_raster",
#     "reproject_raster", "tile_raster", "compute_ndvi", "raster_algebra",
#     "concat_bands", "weighted_band_sum", "zonal_statistics",
#     "reclassify_raster", "focal_statistics", "hillshade", "write_raster",
#     "morans_i", "gearys_c", "gamma_statistic", "getis_ord_g",
#     "moran_local", "getis_ord_g_local", "join_counts_local", "adbscan",
#     "weights_from_shapefile", "distance_band_weights", "knn_weights",
#     "build_transform_and_save_weights", "ols_with_spatial_diagnostics_safe",
#     "build_and_transform_weights", "spatial_markov", "dynamic_lisa", "gm_lag",
#     "create_map", "create_web_map",
#     "download_boundaries", "download_climate_data",
#     "get_species_info", "download_species_occurrences",
#     "download_street_network", "calculate_shortest_path",
#     "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
# }

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# eopssrag branch — single EOPSS GraphRAG system prompt.
# All other prompts commented out; restore on main branch.

_EOPSS_SYSTEM_PROMPT = (
    "You are an expert on the ESA Earth Observation Science Strategy 2040. "
    "You have three tools:\n"
    "- search_eopss(question, k): semantic vector search + knowledge graph traversal; "
    "  use for natural-language questions about the strategy.\n"
    "- query_ontology(cypher): read-only Cypher query on the Memgraph knowledge graph; "
    "  use for structured lookups (e.g. which SQs inform the Paris Agreement with strength ≥ 3?, "
    "  which tipping points are linked to ice-related science questions?).\n"
    "- list_ontology_nodes(label): enumerate nodes by type — available labels: "
    "AreaOfAction, ScienceTheme, ScienceQuestion, StrategicObjective, EnablingTool, "
    "Treaty, NationalPolicy, TippingPoint, Acronym, KnowledgeAdvancementObjective.\n\n"
    "Workflow: always call the appropriate tool first to retrieve context, then synthesize a "
    "precise answer grounded in that context. "
    "Cite specific science question IDs (e.g. SQ-44), ontology node IDs, relationship "
    "strengths, and details where relevant. "
    "Do not answer from memory alone — always ground your answer in retrieved context.\n\n"
    "For greetings or conversational messages, respond briefly without calling any tools."
)

# _EO_STAC_SYSTEM_PROMPT = (...)   # disabled in eopssrag
# _GEE_SYSTEM_PROMPT = (...)       # disabled in eopssrag
# _EVE_RETRIEVAL_SYSTEM_PROMPT = (...)  # disabled in eopssrag
# _RAG_SYSTEM_PROMPT = (...)       # disabled in eopssrag
# _ARXIV_SYSTEM_PROMPT = (...)     # disabled in eopssrag
# _MOOC_SYSTEM_PROMPT = (...)      # disabled in eopssrag
# _GIS_SYSTEM_PROMPT = (...)       # disabled in eopssrag
# _SUPERVISOR_SYSTEM_PROMPT = (...) # disabled in eopssrag

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

# eopssrag branch — single agent talking directly to the GraphRAG MCP server.
eopss_agent = create_agent(
    model,
    tools=[t for t in mcp_tools if t.name in _GRAPHRAG_TOOL_NAMES],
    system_prompt=_EOPSS_SYSTEM_PROMPT,
)

supervisor_agent = eopss_agent  # keeps app.py import working unchanged

# -- Sub-agents disabled in eopssrag branch (restore on main branch) --
# eo_stac_agent       = create_agent(model, tools=[t for t in mcp_tools if t.name in _EO_STAC_TOOL_NAMES], system_prompt=_EO_STAC_SYSTEM_PROMPT)
# gee_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in ["download_satellite_image", "get_lat_lon"]], system_prompt=_GEE_SYSTEM_PROMPT)
# eve_retrieval_agent = create_agent(model, tools=[t for t in mcp_tools if t.name == "retrieve"], system_prompt=_EVE_RETRIEVAL_SYSTEM_PROMPT)
# eoss_rag_agent      = create_agent(model, tools=[t for t in mcp_tools if t.name == "hybrid_rag_retrieve"], system_prompt=_RAG_SYSTEM_PROMPT)
# moocs_agent         = create_agent(model, tools=[t for t in mcp_tools if t.name in _MOOC_TOOL_NAMES], system_prompt=_MOOC_SYSTEM_PROMPT)
# arxiv_agent         = create_agent(model, tools=[t for t in mcp_tools if t.name in _ARXIV_TOOL_NAMES], system_prompt=_ARXIV_SYSTEM_PROMPT)
# gis_agent           = create_agent(model, tools=[t for t in mcp_tools if t.name in _GIS_TOOL_NAMES], system_prompt=_GIS_SYSTEM_PROMPT)
#
# @tool async def get_eo_stac_data(...)  # disabled
# @tool async def get_arxiv_data(...)    # disabled
# @tool async def get_gee_data(...)      # disabled
# @tool async def get_eve_knowledge(...) # disabled
# @tool async def get_hybrid_rag_context(...)  # disabled
# @tool async def get_mooc_context(...)  # disabled
# @tool async def get_gis_analysis(...)  # disabled
#
# supervisor_agent = create_agent(model, tools=[get_eo_stac_data, ...], system_prompt=_SUPERVISOR_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Hot-swap agents without restarting (used by evaluation pipeline)
# ---------------------------------------------------------------------------

def rebuild_agents(model_name: str, provider: str = "ollama", temperature: float = 0.0) -> None:
    """Recreate the EOPSS agent with a new model. MCP tools are reused."""
    global model, eopss_agent, supervisor_agent
    model          = load_llm(model_name, provider, temperature)
    eopss_agent    = create_agent(model, tools=[t for t in mcp_tools if t.name in _GRAPHRAG_TOOL_NAMES], system_prompt=_EOPSS_SYSTEM_PROMPT)
    supervisor_agent = eopss_agent
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
