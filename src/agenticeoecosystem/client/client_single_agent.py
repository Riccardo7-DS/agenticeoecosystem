import asyncio
import logging
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from agenticeoecosystem.config.mcp_config import get_mcp_servers
from agenticeoecosystem.config.credentials import get_langchain_chat_openai

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step log
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
                _step_log.append({"function_name": tc["name"], "arguments": tc["args"],
                                  "tool_call_id": tc.get("id")})
        if getattr(msg, "type", None) == "tool":
            for entry in reversed(_step_log):
                if entry.get("tool_call_id") == getattr(msg, "tool_call_id", None):
                    entry["tool_result"] = msg.content
                    break


# ---------------------------------------------------------------------------
# MCP tool loading
# ---------------------------------------------------------------------------

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
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert Earth Observation and GIS assistant with access to a comprehensive \
set of tools spanning EO dashboards, satellite imagery, spatial analysis, scientific literature, \
and knowledge retrieval.

CAPABILITIES:
- EO Dashboard & STAC: list/search narratives and themes, retrieve narrative content and indicators, \
search STAC collections and items.
- Google Earth Engine: download satellite imagery (Sentinel-2, etc.) for a given bounding box, \
date range, bands, and cloud cover threshold.
- GIS Analysis: geometry operations (buffer, intersect, union, centroid, simplify, etc.), \
vector file I/O (shapefiles, GeoJSON), raster analysis (NDVI, zonal stats, clip, reproject, hillshade), \
coordinate transforms, spatial statistics (Moran's I, LISA, spatial regression), \
map creation (static PNG/PDF or interactive HTML), and geospatial data downloads \
(GADM boundaries, ESA WorldCover, ERA5 climate, OSM street networks, Sentinel-2 NDVI).
- ArXiv: search papers, retrieve abstracts, download full texts, semantic search, citation graphs.
- EVE Knowledge Base: retrieve ESA and Earth Observation background information.
- ESA EO Science Strategy (hybrid RAG): semantic + knowledge-graph retrieval over strategy PDF documents.
- EO MOOCs: search educational course transcripts for EO concepts and methods.

WORKFLOW RULES:
1. For greetings or small talk, respond naturally without calling any tools.
2. Only decline requests that are clearly outside EO, satellite data, GIS, scientific literature, \
or ESA/space topics AND are not conversational. In that case respond: \
"This question is outside the scope of the EO/GIS assistant."
3. Chain tools as needed — pass outputs (e.g. file paths, collection IDs) from one tool into the next.
4. For EO Dashboard queries: load narratives/catalog first, then chain tools to retrieve content or items.
5. For GEE downloads: include absolute file path(s) from the tool result in your response — they are \
required for downstream GIS processing.
6. Prefer get_abstract over download_paper unless the user explicitly requests full text. \
Never download more than one full paper per task.
7. Return results clearly: include titles, themes, URLs, file paths, and any other key metadata.
"""

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def load_llm(model_name: str, provider: str = "eve", temperature: float = 0.0):
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
agent = create_agent(model, tools=mcp_tools, system_prompt=_SYSTEM_PROMPT)


def rebuild_agent(model_name: str, provider: str = "ollama", temperature: float = 0.0) -> None:
    """Recreate the agent with a new model. MCP tools are reused."""
    global model, agent
    model = load_llm(model_name, provider, temperature)
    agent = create_agent(model, tools=mcp_tools, system_prompt=_SYSTEM_PROMPT)
    logger.info("Agent rebuilt — model=%s  provider=%s  temperature=%s", model_name, provider, temperature)


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
        rebuild_agent(args.model, args.provider, args.temperature)

    logger.info("Agent ready")

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
            async for step in agent.astream({"messages": conversation_history}):
                for update in step.values():
                    for message in update.get("messages", []):
                        if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                logger.info("[TOOL CALL] %s — args: %s", tc["name"], tc["args"])
                                print(f"[TOOL CALL] {tc['name']} — args: {tc['args']}")
                        elif message.type == "tool":
                            logger.debug("[TOOL RESULT] %s", message.content)
                            print(message.content)
                        elif message.type == "ai" and message.content:
                            message.pretty_print()
                            logger.info("Final response: %s", message.content)
                            final_ai_content = message.content
            if final_ai_content:
                conversation_history.append({"role": "assistant", "content": final_ai_content})
            _extract_steps_from_messages(conversation_history)

    asyncio.run(run())