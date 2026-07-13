from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from agenticeoecosystem.definitions import ROOT_DIR
from typing import Optional

import logging
logger = logging.getLogger(__name__)

_EO_STAC_SYSTEM_PROMPT =(
    "You are an EO Dashboard and STAC Catalog based assistant. "
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



toolset  = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params = StdioServerParameters(
                             command='python',
                             args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"]),
                             timeout=300))]


def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='eo_stac_agent',
                description='A helpful assistant for answering user questions based on EO Dashboard and STAC Catalog.',
                instruction=_EO_STAC_SYSTEM_PROMPT,
                tools=toolset,
                mode='single_turn'
            )
        else:
            logger.warning("EO STAC agent not initialized due to missing toolset.")
            return None




eo_stac_agent = _fetch_agent(toolset)