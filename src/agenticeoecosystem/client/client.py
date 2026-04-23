import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agenticeoecosystem.config.credentials import get_langchain_chat_openai
from agenticeoecosystem.config.mcp_config import MCP_SERVERS

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step log — populated by get_eo_stac_data / get_gee_data during a pipeline run
# ---------------------------------------------------------------------------

_step_log: list[dict] = []
_token_counts: dict[str, int] = {"input": 0, "output": 0}


def reset_step_log() -> None:
    """Clear step log and token counters before a new pipeline run."""
    _step_log.clear()
    _token_counts["input"] = 0
    _token_counts["output"] = 0


def _extract_steps_from_messages(messages: list) -> None:
    """Parse LangChain message list and append MCP tool calls to _step_log."""
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            _token_counts["input"] += msg.usage_metadata.get("input_tokens", 0)
            _token_counts["output"] += msg.usage_metadata.get("output_tokens", 0)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                _step_log.append({"function_name": tc["name"], "arguments": tc["args"]})


async def load_tools_from_mcp():

    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    tools = await mcp_client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    return tools

import nest_asyncio
nest_asyncio.apply()
mcp_tools = asyncio.get_event_loop().run_until_complete(load_tools_from_mcp())

# ---------------------------------------------------------------------------
# System prompts (extracted so rebuild_agents can reuse them)
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
    "REQUIRED FORMAT RULES:\n"
    "- start_date: string (YYYY-MM-DD)\n"
    "- end_date: string (YYYY-MM-DD)\n"
    "- bounding_box: [min_lon, min_lat, max_lon, max_lat]\n"
    "- bands: MUST be a comma-separated string (e.g., \"B4,B3,B2,B8,B11\") — NOT a list\n"
    "- scale: integer\n"
    "- image_count: integer\n"
    "- max_cloud_cover: integer\n"
    "Extract all other parameters like dates, cloud cover, and image count from the query."
)

_SUPERVISOR_SYSTEM_PROMPT = (
    "You are a routing agent. "
    "You can get data from EO Dashboards and also Google Earth Engine. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def load_llm(model_name: str, provider: str = "eve", temperature: float = 0.0):
    """
    Build a LangChain chat model.

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
        # model_name can be a registry key ("qwen2.5-14b") or a raw HF model ID
        key = model_name if model_name in MODEL_REGISTRY else None
        if key:
            return get_vllm_langchain_model(key, host=host, port=port, temperature=temperature)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="EMPTY",
            model=model_name,
            temperature=temperature,
        )
    return init_chat_model(model_name, model_provider=provider, temperature=temperature)


# ---------------------------------------------------------------------------
# Agent builder — call rebuild_agents() to switch models without restarting
# ---------------------------------------------------------------------------

_DEFAULT_MODEL: str = os.getenv("MAIN_MODEL_NAME", "EVE-Instruct")
_DEFAULT_PROVIDER: str = os.getenv("MODEL_PROVIDER", "eve")

model = load_llm(_DEFAULT_MODEL, _DEFAULT_PROVIDER)

eo_stac_agent = create_agent(model, tools=mcp_tools[:8], system_prompt=_EO_STAC_SYSTEM_PROMPT)
gee_agent     = create_agent(model, tools=mcp_tools[8:9], system_prompt=_GEE_SYSTEM_PROMPT)


@tool
async def get_eo_stac_data(request: str) -> str:
    """Get data from EO Dashboard and STAC catalogs.
    This handles cross platform functionalities, for examlple if the query is fetch the stac collection data about any narrative, it will first
    search for narrative, pick one and parse the collection code, use the code to get the stac collection items.
    """
    print(f"[TOOL CALL] get_eo_stac_data with request: {request}")

    result = await eo_stac_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    _extract_steps_from_messages(result["messages"])
    text_output = result["messages"][-1].text
    if isinstance(text_output, tuple):
        text_output = " ".join(str(t) for t in text_output)
    print(f"[TOOL RESULT] {text_output}")
    return str(text_output)

@tool
async def get_gee_data(request: str) -> str:
    """Phrase your request in the following way :
     Please download the satellite image for a bounding box defined by the coordinates 
    [52.526831501, 24.130796795, 52.578119029, 24.181873633]. The desired date range is 2020-01-01 to 2021-01-31.
    The scale is 10 meters. Split the region into smaller tiles if necessary but download maximum 3 images.
    Select the bands B4, B3, B2, B8, B11.
    Filter for less than 20% cloud cover and download 3 image
    REQUIRED FORMAT RULES:
- start_date: string (YYYY-MM-DD)
- end_date: string (YYYY-MM-DD)
- bounding_box: [min_lon, min_lat, max_lon, max_lat]
- bands: MUST be a comma-separated string (e.g., "B4,B3,B2,B8,B11") — NOT a list
- scale: integer
- image_count: integer
- max_cloud_cover: integer
    """
    print(f"[TOOL CALL] get_eo_stac_data with request: {request}")
    result = await gee_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    _extract_steps_from_messages(result["messages"])
    text_output = result["messages"][-1].text
    if isinstance(text_output, tuple):
        text_output = " ".join(str(t) for t in text_output)
    return str(text_output)

supervisor_agent = create_agent(model, tools=[get_eo_stac_data, get_gee_data], system_prompt=_SUPERVISOR_SYSTEM_PROMPT)


def rebuild_agents(model_name: str, provider: str = "ollama", temperature: float = 0.0) -> None:
    """Recreate all agents with a new model without restarting the process.

    MCP tools are reused (loaded once at import). Only the LangChain agents
    are rebuilt, which is cheap.

    Args:
        model_name  : Model identifier (e.g. "nemotron-cascade-2:latest", "claude-sonnet-4-6").
        provider    : LangChain model provider (e.g. "ollama", "anthropic") or "eve".
        temperature : Sampling temperature.
    """
    global model, eo_stac_agent, gee_agent, supervisor_agent
    model         = load_llm(model_name, provider, temperature)
    eo_stac_agent = create_agent(model, tools=mcp_tools[:8],  system_prompt=_EO_STAC_SYSTEM_PROMPT)
    gee_agent     = create_agent(model, tools=mcp_tools[8:9], system_prompt=_GEE_SYSTEM_PROMPT)
    supervisor_agent = create_agent(model, tools=[get_eo_stac_data, get_gee_data], system_prompt=_SUPERVISOR_SYSTEM_PROMPT)
    logger.info("Agents rebuilt with model=%s provider=%s temperature=%s", model_name, provider, temperature)

if __name__ == "__main__":
    async def run():
        user_request = (
            "Find any narrative about the atmosphere, pick one and get an image from the GEE of the location related to the narrative"
        )
        print("User Request:", user_request)
        print("\n" + "="*80 + "\n")

        async for step in supervisor_agent.astream(
            {"messages": [{"role": "user", "content": user_request}]}
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    # print(f"Message type : {message.type}")
                    if message.type == "tool":
                        continue
                    elif message.type == "ai":
                        message.pretty_print()
    asyncio.run(run())