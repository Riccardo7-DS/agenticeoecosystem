from langchain_openai import ChatOpenAI
import base64
from langchain_mcp_adapters.client import MultiServerMCPClient
from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import asyncio
import logging
from dotenv import load_dotenv
from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages, AIMessage

# from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

import os
username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
credentials = base64.b64encode(f"{username}:{password}".encode()).decode() 




logger = logging.getLogger(__name__)
load_dotenv()
async def load_tools_from_mcp():

    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    tools = await mcp_client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    return tools

mcp_tools = asyncio.run(load_tools_from_mcp())

def load_llm(eve=False):
    # Declare the LLM with the EVE model and credentials
    if eve:
        return ChatOpenAI(
            base_url=os.getenv("EVE_ENDPOINT"),
            api_key="EMPTY",
            model=os.getenv("MAIN_MODEL_NAME"),
                    temperature=0,
                    default_headers={
            "Authorization": f"Basic {credentials}"
        })
    else:
        llm = init_chat_model('nemotron-cascade-2:latest', model_provider="ollama", temperature=0)
        return llm

model = load_llm()

eo_stac_agent = create_agent(
    model,
    tools=mcp_tools[:8],
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
    )
)

gee_agent = create_agent(
    model,
    tools=mcp_tools[8:9],
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
    )
)


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
    # return result["messages"][-1].text
    text_output = result["messages"][-1].text
    if isinstance(text_output, tuple):
        text_output = " ".join(str(t) for t in text_output)
    print(f"[TOOL RESULT] {text_output}")
    # logger.info(result)
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
    text_output = result["messages"][-1].text
    if isinstance(text_output, tuple):
        text_output = " ".join(str(t) for t in text_output)
    return str(text_output)
    # return result["messages"][-1].text

supervisor_agent = create_agent(
    model,
    tools=[get_eo_stac_data, get_gee_data],
    system_prompt=(
        "You are a routing agent. "
        "You can get data from EO Dashboards and also Google Earth Engine. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    )
)

if __name__ == "__main__":
    # Example: User request requiring both calendar and email coordination
    async def run():
        # user_request = (
        #     "Fetch the stac collection items for any narrative related to covid-19. ."
        # )

        # user_request = (
        #     "Find the narrative about covid-19 impact on automobile industry and fetch its stac collection items.​"
        # )

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