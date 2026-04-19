import argparse
import asyncio
import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from agenticeoecosystem.config.mcp_config import get_mcp_servers
from agenticeoecosystem.utils import setup_logging

setup_logging("client")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger.info("Loading .env from %s", PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
if not username or not password:
    raise ValueError("EVE_USERNAME and EVE_PASSWORD env vars not set (check .env)")
credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
logger.info("Loaded EVE credentials from environment")

class _EVEChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that normalises tool-message content to plain strings.

    The EVE API is stricter than OpenAI: it rejects list-typed content and
    extra fields (e.g. LangChain's internal 'id') in tool messages.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for msg in payload.get("messages", []):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
                msg["content"] = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in msg["content"]
                )
        return payload

async def load_tools_from_mcp():
    logger.info("Building MCP server configuration")
    mcp_servers = get_mcp_servers()
    logger.info("MCP servers configured: %s", list(mcp_servers.keys()))
    mcp_client = MultiServerMCPClient(mcp_servers)
    logger.info("Requesting tools from MCP servers")
    tools = await mcp_client.get_tools()
    logger.info("MCP tool discovery complete: %d tools", len(tools))
    for mcp_tool in tools:
        logger.info("  - %s: %s", mcp_tool.name, mcp_tool.description)
    return tools


logger.info("Starting MCP tool bootstrap")
mcp_tools = asyncio.run(load_tools_from_mcp())
logger.info("MCP tool bootstrap finished")


def load_llm(eve=True):
    logger.info("Initializing LLM (eve=%s)", eve)
    if eve:
        return _EVEChatOpenAI(
            base_url=os.getenv("EVE_ENDPOINT"),
            api_key="EMPTY",
            model=os.getenv("MAIN_MODEL_NAME"),
                    temperature=0,
                    default_headers={
            "Authorization": f"Basic {credentials}"
        })
    else:
        llm = init_chat_model(
            "mistral",
            model_provider="ollama",
            temperature=0,
        )
        return llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgenticEO client")
    parser.add_argument(
        "--llm",
        choices=["eve", "ollama"],
        default="ollama",
        help="LLM backend: 'eve' or 'ollama' (default: ollama)",
    )
    args = parser.parse_args()
    use_eve = args.llm == "eve"

    model = load_llm(eve=use_eve)
    logger.info("LLM initialization complete")

    _EO_STAC_TOOLS = {
        "list_narratives", "search_narratives", "get_narrative_content",
        "list_themes", "get_narrative_indicators", "search_collections",
        "get_collection_items", "find_narratives_for_indicator",
    }
    eo_stac_agent = create_agent(
        model,
        tools=[t for t in mcp_tools if t.name in _EO_STAC_TOOLS],
        system_prompt=(
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
        ),
    )

    gee_agent = create_agent(
        model,
        tools=[t for t in mcp_tools if t.name == "download_satellite_image"],
        system_prompt=(
            """You are an expert GEE assistant responsible for translating user requests into precise tool calls.
	Your task is to answer the user's query about satellite image downloads by using the `download_satellite_image` tool.
            REQUIRED FORMAT RULES:
            - start_date: string (YYYY-MM-DD)
            - end_date: string (YYYY-MM-DD)
            - bounding_box: [min_lon, min_lat, max_lon, max_lat]
            - bands: MUST be a comma-separated string (e.g., "B4,B3,B2,B8,B11") — NOT a list
            - scale: integer
            - image_count: integer
            - max_cloud_cover: integer

	Extract all other parameters like dates, cloud cover, and image count from the query."""
        ),
    )

    eve_retrieval_agent = create_agent(
        model,
        tools=[t for t in mcp_tools if t.name == "retrieve"],
        system_prompt=(
            "You are a knowledge retrieval assistant for Earth Observation and ESA topics. "
            "Use the retrieve tool to find relevant information and answer the user's question."
        ),
    )

    rag_agent = create_agent(
        model,
        tools=[t for t in mcp_tools if t.name == "hybrid_rag_retrieve"],
        system_prompt=(
            "You are a retrieval assistant for ESA Earth Observation Science Strategy. "
            "Use the hybrid_rag_retrieve tool to fetch relevant document chunks and "
            "knowledge graph context, then return the raw context to the caller."
        ),
    )

    @tool
    async def get_eo_stac_data(request: str) -> str:
        """Get data from EO Dashboard and STAC catalogs.
        This handles cross platform functionalities, for examlple if the query is fetch the stac collection data about any narrative, it will first
        search for narrative, pick one and parse the collection code, use the code to get the stac collection items.
        """
        logger.info("[TOOL CALL] get_eo_stac_data: %s", request)
        result = await eo_stac_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        # logger.info("[TOOL RESULT] %s", text_output)
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
        logger.info("[TOOL CALL] get_gee_data: %s", request)
        result = await gee_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)

    @tool
    async def get_eve_knowledge(request: str) -> str:
        """Retrieve knowledge about ESA and Earth Observation from the EVE knowledge base."""
        result = await eve_retrieval_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        return str(text_output)

    @tool
    async def get_hybrid_rag_context(request: str) -> str:
        """Retrieve relevant context from the ESA EO Science Strategy knowledge base.

        Uses hybrid retrieval (semantic vector search + ESA knowledge graph) over
        PDF documents. Use for questions about science themes, science questions,
        strategic objectives, enabling tools, tipping points, treaties, and other
        ESA EO domain topics.
        """
        logger.info("[TOOL CALL] get_hybrid_rag_context: %s", request)
        result = await rag_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        for msg in result["messages"]:
            for tc in getattr(msg, "tool_calls", []):
                logger.info("[rag_agent -> hybrid_rag_retrieve] question: %s", tc.get("args", {}))
            if getattr(msg, "type", None) == "tool":
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
                logger.info("[hybrid_rag_retrieve raw output]\n%s", content)
        text_output = result["messages"][-1].text
        if isinstance(text_output, tuple):
            text_output = " ".join(str(t) for t in text_output)
        # logger.info("[TOOL RESULT] %s", text_output)
        return str(text_output)

    supervisor_agent = create_agent(
        model,
        tools=[get_eo_stac_data, get_gee_data, get_eve_knowledge, get_hybrid_rag_context],
        system_prompt=(
            "You are a routing agent. "
            "You can get data from EO Dashboards, Google Earth Engine, "
            "the EVE knowledge base for ESA/EO background information, "
            "and the ESA EO Science Strategy knowledge base (hybrid RAG). "
            "Break down user requests into appropriate tool calls and coordinate the results. "
            "When a request involves multiple actions, use multiple tools in sequence. "
            "IMPORTANT: Return the tool result directly and completely to the user, word for word. "
            "Do not add introductions, summaries, or commentary around tool results. "
        ),
    )

    logger.info("All agents ready")

    async def run():
        logger.info("Entering interactive loop")
        while True:
            user_request = input("\nQuery (or 'quit' to exit): ").strip()
            if not user_request:
                continue
            if user_request.lower() in ("quit", "exit", "q"):
                break
            logger.info("User query: %s", user_request)
            print("=" * 80)
            async for step in supervisor_agent.astream(
                {"messages": [{"role": "user", "content": user_request}]}
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        if message.type == "tool":
                            print(message.content)
                            logger.info("Tool result: %s", message.content)
                        elif message.type == "ai" and message.content:
                            message.pretty_print()
                            logger.info("Final response: %s", message.content)

    asyncio.run(run())
