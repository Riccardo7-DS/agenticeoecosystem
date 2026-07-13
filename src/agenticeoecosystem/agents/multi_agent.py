# from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
# apply_eve_patch()

from google.adk.agents.llm_agent import Agent

from google.adk.models.lite_llm import LiteLlm
import base64
import logging
import os
from agenticeoecosystem.definitions import ROOT_DIR
from dotenv import load_dotenv
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from litellm import query
from mcp import StdioServerParameters
import asyncio
import json
import uuid
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

from agenticeoecosystem.agents.subagents import ALL_SUB_AGENTS, llm


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

load_dotenv()
logger = logging.getLogger(__name__)



# _SUPERVISOR_SYSTEM_PROMPT = (
#     "You are a routing agent for an Earth Observation / GIS assistant. "
#     "You coordinate access to:\n"
#     "- EO Dashboard narratives and STAC catalog data\n"
#     "- Satellite imagery from CDSE (Copernicus Data Space Ecosystem)\n"
#     "- EO MOOC (online course) transcripts\n"
#     "- ArXiv scientific papers\n"
#     # "- EVE knowledge base for ESA/EO background information\n"
#     "- ESA EO Science Strategy knowledge base (Graph RAG)\n"
#     "- GIS analysis server for spatial operations, raster analysis, visualization, and geospatial downloads\n"
#     "\n"
#     "ROUTING:\n"
#     "- Break requests into the appropriate tool call(s) and coordinate results.\n"
#     "- If a request needs multiple sources, call tools in sequence and combine their outputs.\n"
#     "\n"
#     "OUTPUT RULES:\n"
#     "- Return tool results directly and completely, word for word — no added "
#     "introductions, summaries, or commentary.\n"
#     "- Never turn a local image path into a hyperlink or markdown link; return "
#     "the raw local path as plain text.\n"
#     "\n"
#     "CONVERSATION: For greetings, thanks, or small talk, respond naturally and "
#     "briefly without calling any tools.\n"
#     "\n"
#     "SCOPE: Only decline requests that are both (a) unrelated to Earth "
#     "Observation, satellite data, GIS, scientific literature, or ESA/space "
#     "topics, and (b) not conversational (e.g. restaurant recommendations, "
#     "financial trading, cooking, sports scores). For those, call no tools and "
#     "respond exactly: 'This question is outside the scope of the EO/GIS "
#     "assistant.'"
# )
_SUPERVISOR_SYSTEM_PROMPT = (
    "You are a routing agent for an Earth Observation / GIS assistant. "
"You coordinate access to:\n"
"- EO Dashboard narratives and STAC catalog data\n"
"- Satellite imagery from CDSE (Copernicus Data Space Ecosystem)\n"
"- EO MOOC (online course) transcripts\n"
"- ArXiv scientific papers\n"
# "- EVE knowledge base for ESA/EO background information\n"
"- ESA EO Science Strategy knowledge base (Graph RAG)\n"
"- GIS analysis server for spatial operations, raster analysis, visualization, and geospatial downloads\n"
"- GEE (Google Earth Engine) satellite imagery sub-agent, for downloading "
"specialized collections not available via CDSE (e.g. VIIRS, MODIS, or "
"other GEE-only datasets) for a given place, coordinate, bounding box, or "
"region over a date range\n"
"- ERA5 climate sub-agent, for downloading and inspecting ERA5 reanalysis "
"data (pressure-level or surface-level variables like temperature, "
"geopotential, precipitation) as NetCDF files for a given year/month\n"
"\n"
"ROUTING:\n"
"- Break requests into the appropriate tool call(s) and coordinate results.\n"
"- If a request needs multiple sources, call tools in sequence and combine their outputs.\n"
"- For general or 'sample' imagery requests (e.g. Sentinel-2, Sentinel-1, "
"Landsat true-color images of a place), use the CDSE sub-agent by default.\n"
"- Only use the GEE sub-agent when the requested collection is NOT "
"available through CDSE — e.g. VIIRS, MODIS, or other specialized/GEE-only "
"datasets. Do not use GEE for imagery CDSE already covers.\n"
"- Use the ERA5 sub-agent, as usual, for atmospheric/climate reanalysis "
"variables (temperature, wind, geopotential, precipitation, etc.) rather "
"than imagery.\n"
"\n"
"OUTPUT RULES:\n"
"- Return tool results directly and completely, word for word — no added "
"introductions, summaries, or commentary.\n"
"- Never turn a local image path into a hyperlink or markdown link; return "
"the raw local path as plain text.\n"
"\n"
"CONVERSATION: For greetings, thanks, or small talk, respond naturally and "
"briefly without calling any tools.\n"
"\n"
"SCOPE: Only decline requests that are both (a) unrelated to Earth "
"Observation, satellite data, GIS, scientific literature, or ESA/space "
"topics, and (b) not conversational (e.g. restaurant recommendations, "
"financial trading, cooking, sports scores). For those, call no tools and "
"respond exactly: 'This question is outside the scope of the EO/GIS "
"assistant.'"
)

root = Agent(
    model=llm,
    name="orchestrator_agent",      # coordinator agent
    sub_agents=ALL_SUB_AGENTS, # List all sub-agents here
    description="An orchestrator agent that routes user queries to its sub-agents based on the query content.",
    instruction=_SUPERVISOR_SYSTEM_PROMPT
# Auto-injects: request_task_weather_checker, request_task_flight_booker
)


APP_NAME = "EVE-AGENT-ECOSYSTEM"
USER_ID = "user_1"
SESSION_ID = "session_001" # Using a fixed ID for simplicity

session_service = InMemorySessionService()

async def init_session(app_name:str,user_id:str,session_id:str) -> InMemorySessionService:
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
    return session

session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

runner = Runner(agent=root, # The agent we want to run
                app_name=APP_NAME,   # Associates runs with our app
                session_service=session_service # Uses our session manager
                )
print(f"Runner created for agent '{runner.agent.name}'.")

import json
import uuid

from google.genai import types  # adjust import to match your actual SDK path


def _extract_final_text(event) -> str | None:
    """
    Pull plain text out of the final response event, if any.
    Guards against parts that are function_call / function_response
    (which don't have a usable `.text`).
    """
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

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # --- Print which agent is currently active ---
        if event.author:
            print(f"\n[AGENT] {event.author}")

        # --- Print tool calls / tool results ---
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    print(f"  [TOOL CALL] {fc.name}({dict(fc.args)})")
                elif hasattr(part, 'function_response') and part.function_response:
                    fr = part.function_response
                    # Truncate long responses for readability
                    response_str = str(fr.response)
                    if len(response_str) > 300:
                        response_str = response_str[:300] + "... [truncated]"
                    print(f"  [TOOL RESULT] {fr.name} → {response_str}")

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_response_text = text
            elif event.actions and event.actions.escalate:
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    print(f"\n<<< Agent Response: {final_response_text}")


async def run_conversation():
    await call_agent_async(
        "Get me two papers on sea-ice from arxiv, also fetch me similar "
        "information from any course material you have access to.",
        runner, USER_ID, SESSION_ID,
    )


async def get_ans_tool_path(query: str) -> str:
    """
    Runs a query through the orchestrator in an isolated session and returns
    a JSON string with the final answer plus the sequence of agents and
    tools invoked to produce it.
    """
    session_id = f"session_{uuid.uuid4().hex}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_answer = "Agent did not produce a final response."
    agents_called = []
    tools_called = []

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.author and (not agents_called or agents_called[-1] != event.author):
            agents_called.append(event.author)

        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, 'function_call', None):
                    tools_called.append(part.function_call.name)

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_answer = text
            elif event.actions and event.actions.escalate:
                final_answer = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    return json.dumps({
        "final_answer": final_answer,
        "tools_called": tools_called,
        "agents_called": agents_called,
    })
if __name__ == "__main__":
    try:
        # ans = asyncio.run(get_ans_tool_path("Download the geopotential height at 500 hPa for January 2023, and tell me what the file looks like once it's downloaded."))
        ans = asyncio.run(get_ans_tool_path("Download a VIIRS image over Naples, Italy for the first week of March 2024."))
        print(ans)
    except Exception as e:
        print(f"An error occurred: {e}")    