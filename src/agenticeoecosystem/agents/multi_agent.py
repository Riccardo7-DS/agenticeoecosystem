from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
apply_eve_patch()

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

# import litellm
# litellm._turn_on_debug()

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

load_dotenv()
logger = logging.getLogger(__name__)



# _SUPERVISOR_SYSTEM_PROMPT = (
#     "You are a routing agent for an Earth Observation / GIS assistant. "
# "You coordinate access to:\n"
# "- EO Dashboard narratives and STAC catalog data\n"
# "- Satellite imagery from CDSE (Copernicus Data Space Ecosystem)\n"
# "- EO MOOC (online course) transcripts\n"
# "- ArXiv scientific papers\n"
# "- ESA EO Science Strategy knowledge base (Graph RAG)\n"
# "- GIS analysis server for spatial operations, raster analysis, visualization\n"
# "- GEE (Google Earth Engine) satellite imagery sub-agent, for downloading "
# "specialized collections not available via CDSE (e.g. VIIRS, MODIS, or "
# "other GEE-only datasets) for a given place, coordinate, bounding box, or "
# "region over a date range\n"
# "- ERA5 climate sub-agent, for downloading and inspecting ERA5 reanalysis "
# "data (pressure-level or surface-level variables like temperature, "
# "geopotential, precipitation) as NetCDF files for a given year/month\n"
# "\n"
# "ROUTING:\n"
# "- Break requests into the appropriate tool call(s) and coordinate results.\n"
# "- If a request needs multiple sources, call tools in sequence and combine their outputs.\n"
# "- For general or 'sample' imagery requests (e.g. Sentinel-2, Sentinel-1, "
# "Landsat true-color images of a place), use the CDSE sub-agent by default.\n"
# "- Only use the GEE sub-agent when the requested collection is NOT "
# "available through CDSE — e.g. VIIRS, MODIS, or other specialized/GEE-only "
# "datasets. Do not use GEE for imagery CDSE already covers.\n"
# "- Use the ERA5 sub-agent, as usual, for atmospheric/climate reanalysis "
# "variables (temperature, wind, geopotential, precipitation, etc.) rather "
# "than imagery.\n"
# "\n"
# "OUTPUT RULES:\n"
# "- Return tool results directly and completely, word for word — no added "
# "introductions, summaries, or commentary.\n"
# "- Never turn a local image path into a hyperlink or markdown link; return "
# "the raw local path as plain text.\n"
# "\n"
# "CONVERSATION: For greetings, thanks, or small talk, respond naturally and "
# "briefly without calling any tools.\n"
# "\n"
# "SCOPE: Only decline requests that are both (a) unrelated to Earth "
# "Observation, satellite data, GIS, scientific literature, or ESA/space "
# "topics, and (b) not conversational (e.g. restaurant recommendations, "
# "financial trading, cooking, sports scores). For those, call no tools and "
# "respond exactly: 'This question is outside the scope of the EO/GIS "
# "assistant.'"
# )

_SUPERVISOR_SYSTEM_PROMPT = (
    "You are a routing agent for an Earth Observation / GIS assistant. "
    "You coordinate access to:\n"
    "- EO Dashboard narratives and STAC catalog data\n"
    "- Satellite quicklook imagery from CDSE (Copernicus Data Space Ecosystem): "
    "Sentinel-1/2/3, Sentinel-5P, Landsat, DEM — plus place name <-> bbox geocoding\n"
    "- EUMETSAT geostationary product search & download: MTG FCI and MSG SEVIRI "
    "raw/tailored Data Store products only\n"
    "- EO MOOC (online course) transcripts\n"
    "- ArXiv scientific papers\n"
    "- ESA EO Science Strategy knowledge base (Graph RAG)\n"
    "- GIS analysis server for spatial operations, raster analysis, visualization\n"
    "- GEE (Google Earth Engine) satellite imagery sub-agent, for downloading "
    "specialized collections not available via CDSE (e.g. VIIRS, MODIS, or "
    "other GEE-only datasets) for a given place, coordinate, bounding box, or "
    "region over a date range\n"
    "- ERA5 climate sub-agent, for downloading and inspecting ERA5 reanalysis "
    "data (pressure-level or surface-level variables like temperature, "
    "geopotential, precipitation) as NetCDF files for a given year/month\n"
    "\n"
    "ROUTING:\n"
    "- Every sub-agent above is called directly as a function tool by its own "
    "name (e.g. cdse_agent(request=...), climate_agent(request=...)). There is "
    "no transfer_to_agent tool in this setup — never call a tool named "
    "transfer_to_agent; it does not exist here and calling it will fail.\n"
    "- Break requests into the appropriate tool call(s) and coordinate results.\n"
    "- If a request needs multiple sources, call tools in sequence and combine their outputs.\n"
    "- For general or 'sample' imagery requests (e.g. Sentinel-2, Sentinel-1, "
    "Landsat true-color images of a place), use the CDSE sub-agent by default.\n"
    "- Use the EUMETSAT sub-agent only for geostationary MTG FCI or MSG SEVIRI "
    "products — not for Sentinel/Landsat/DEM quicklooks, which stay on CDSE. "
    "EUMETSAT cannot geocode place names itself: if the user gives a place name "
    "rather than a bbox, geocode it via CDSE first and pass the resulting bbox "
    "to EUMETSAT.\n"
    "- Only use the GEE sub-agent when the requested collection is NOT "
    "available through CDSE — e.g. VIIRS, MODIS, or other specialized/GEE-only "
    "datasets. Do not use GEE for imagery CDSE or EUMETSAT already covers.\n"
    "- Use the ERA5 sub-agent, as usual, for atmospheric/climate reanalysis "
    "variables (temperature, wind, geopotential, precipitation, etc.) rather "
    "than imagery.\n"
    "- EUMETSAT downloads (MTG FCI, MSG SEVIRI) are raw geostationary formats "
    "(.nat, segmented HRIT, ...) that the GIS analysis server cannot open "
    "directly — never hand a raw EUMETSAT file path to the GIS sub-agent. "
    "The EUMETSAT sub-agent reads/processes its own downloads internally "
    "(band reflectance/brightness-temperature stats, calibration, GeoTIFF "
    "export) — route any 'read/inspect/report values from' request on a "
    "EUMETSAT file back to the EUMETSAT sub-agent, not to GIS. Only pass a "
    "EUMETSAT product to the GIS sub-agent for genuinely custom GIS "
    "operations (NDVI, zonal stats, reprojecting to a different CRS than "
    "geographic, ...), and only after the EUMETSAT sub-agent has exported "
    "it to GeoTIFF. The EUMETSAT sub-agent is single-turn and has no memory "
    "of a previous call — if a request needs both a download and a read of "
    "the same file, put both in the SAME eumetsat_agent call (e.g. 'download "
    "X and report the reflectance range'), not two separate calls; a "
    "second call describing a file path from a prior call's response will "
    "not work.\n"
    "- CDSE downloads (Sentinel/Landsat/DEM quicklooks) are already standard "
    "georeferenced rasters (GeoTIFF/COG) — for reprojection, clipping, NDVI, "
    "zonal stats, or any other spatial processing on them, route to the GIS "
    "analysis server as usual; CDSE does not do this itself.\n"
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

def _on_tool_error(tool, args, tool_context, error):
    """Turn a tool-execution failure into a normal tool result instead of
    letting it propagate and abort the turn (which otherwise leaves that
    tool call's id permanently unresolved in history — e.g. when the model
    hallucinates a call to a tool that isn't registered, such as
    transfer_to_agent, which this orchestrator never exposes)."""
    logger.warning("Tool '%s' failed: %s", getattr(tool, "name", "?"), error)
    return {"error": f"Tool '{getattr(tool, 'name', '?')}' failed: {error}"}


root = Agent(
    model=llm,
    name="orchestrator_agent",      # coordinator agent
    sub_agents=ALL_SUB_AGENTS, # List all sub-agents here
    description="An orchestrator agent that routes user queries to its sub-agents based on the query content.",
    instruction=_SUPERVISOR_SYSTEM_PROMPT,
    on_tool_error_callback=_on_tool_error,
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
        query = input("Enter your question: ")
        ans = asyncio.run(get_ans_tool_path(query))
        print(ans)
    except Exception as e:
        print(f"An error occurred: {e}")
