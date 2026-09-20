from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
apply_eve_patch()

from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as apply_mcp_image_patch
apply_mcp_image_patch()

from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard as _orphan_guard
_orphan_guard.apply()

from agenticeoecosystem.agents.subagents import arxiv_rate_limit_retry as _arxiv_retry
_arxiv_retry.apply()

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
    "specialized collections not available via CDSE (e.g. NASA products like "
    "VIIRS or MODIS, JAXA products like GPM/GSMaP or ALOS, or other GEE-only "
    "datasets) for a given place, coordinate, bounding box, or region over a "
    "date range\n"
    "- ERA5 climate sub-agent, for downloading and inspecting ERA5 reanalysis "
    "data (pressure-level or surface-level variables like temperature, "
    "geopotential, precipitation) as NetCDF files for a given year/month\n"
    "- VLM (Vision-Language Model) sub-agent, for visually inspecting and "
    "reasoning over image files already on disk — e.g. a downloaded "
    "quicklook, a single satellite scene, or a directory of images from a "
    "prior step (up to 10 images at a time) — to answer questions like "
    "'what does this look like', 'is there visible cloud cover in these "
    "images', or 'describe what changed between these scenes'\n"
    "- TLE/orbit sub-agent, for computing where a satellite (or set of "
    "satellites) is at a given UTC time, and for predicting upcoming "
    "overpasses of a target location — from TLE data (Celestrak/"
    "Space-Track) via SGP4 propagation. Does not fetch or analyze imagery.\n"
    "\n"
    "ROUTING:\n"
    "- Break requests into the appropriate tool call(s) and coordinate results.\n"
    "- If a request needs multiple sources, call tools in sequence and combine their outputs.\n"
    "- When a request contains multiple clauses (e.g. 'do X, then use the "
    "result to do Y'), identify all clauses before acting, and do not "
    "produce a final answer until every clause has been addressed by an "
    "appropriate sub-agent call. A sub-agent's response to only the first "
    "clause is not a complete answer — continue routing to the next "
    "required sub-agent.\n"
    "- NEVER issue two calls to the SAME sub-agent in parallel/in the same "
    "turn, even for genuinely different requests (e.g. a 'before' and "
    "'after' image from the same source). Wait for the first call to that "
    "sub-agent to fully return before issuing the second — concurrent calls "
    "to the same sub-agent can cross-contaminate each other's request "
    "context (e.g. both ending up using the second call's date range). "
    "This restriction is specific to repeated calls to the same sub-agent; "
    "calls to different sub-agents may still run in sequence as usual.\n"
    "- For general or 'sample' imagery requests (e.g. Sentinel-2, Sentinel-1, "
    "Landsat true-color images of a place), use the CDSE sub-agent by default.\n"
    "- Use the EUMETSAT sub-agent only for geostationary MTG FCI or MSG SEVIRI "
    "products — not for Sentinel/Landsat/DEM quicklooks, which stay on CDSE. "
    "EUMETSAT cannot geocode place names itself: if the user gives a place name "
    "rather than a bbox, geocode it via CDSE first and pass the resulting bbox "
    "to EUMETSAT.\n"
    "- Only use the GEE sub-agent when the requested collection is NOT "
    "available through CDSE — e.g. NASA products (VIIRS, MODIS), JAXA "
    "products (GPM/GSMaP precipitation, ALOS), or other specialized/GEE-only "
    "datasets. Do not use GEE for imagery CDSE or EUMETSAT already covers.\n"
    "- Use the ERA5 sub-agent, as usual, for atmospheric/climate reanalysis "
    "variables (temperature, wind, geopotential, precipitation, etc.) rather "
    "than imagery.\n"
    "- Use the VLM sub-agent whenever the request requires actually looking "
    "at image content rather than just fetching or processing it — e.g. "
    "visual description, qualitative comparison between scenes, spotting "
    "clouds/smoke/flooding by eye, or any question about what an image "
    "'shows'. First obtain the image(s) via the relevant data source "
    "(CDSE, EUMETSAT, GEE, GIS analysis server, etc.), then pass the "
    "resulting local file or directory path and the user's question to the "
    "VLM sub-agent. If there are multiple images to compare, pass them one by one "
    "to the VLM sub-agent. Do not use the VLM sub-agent for quantitative raster "
    "analysis (NDVI, zonal stats, band math) — that stays on the GIS "
    "analysis server.\n"
    "- The VLM sub-agent can apply deeper domain-specific visual criteria on "
    "request. If the request concerns detecting or assessing a wildfire, "
    "active fire, smoke plume, or burned/burn-scarred area in the image(s), "
    "start your handoff instruction to the VLM sub-agent with the line "
    "`VLM_DOMAIN: wildfire` on its own line, then your usual instruction — "
    "this makes it apply wildfire-specific visual criteria. Omit that line "
    "for every other visual-analysis request (it should stay generic by "
    "default).\n"
    "- EUMETSAT and CDSE do not do NDVI, reprojection, clipping, or zonal "
    "stats on downloaded files — route that to the GIS analysis server.\n"
    "- After the VLM sub-agent reports a finding tied to a location (e.g. a "
    "detected wildfire/smoke plume with an approximate lat/lon), use the "
    "TLE sub-agent's get_acquisition_schedule to predict which satellites "
    "will next pass over that location and when, then combine both results "
    "into a recommended acquisition/monitoring plan (which platform, next "
    "pass time, priority) as the final answer — do not just report the VLM "
    "finding alone when the user asked for a plan or next steps. If the "
    "user names specific satellites, pass them explicitly; otherwise let "
    "the tool use its default EO constellation. get_satellite_position/"
    "get_positions_for_set are for simple 'where is X right now/at time T' "
    "questions with no target location.\n"
    "- Only route to the GIS analysis server when a raster/vector file has "
    "already been produced earlier in THIS conversation (e.g. by CDSE, GEE, "
    "EUMETSAT, or ERA5) — never call a GIS tool (clip, reproject, zonal "
    "stats, band math, etc.) on a file path that no prior tool result "
    "actually returned. For a request that is purely a narrative/text "
    "search (e.g. EO Dashboard, arXiv, MOOCs, ESA EOSS knowledge base), do "
    "not invoke the GIS sub-agent at all.\n"
    "- Before clipping a raster to a boundary (e.g. an administrative "
    "region's shapefile/GeoJSON), make sure that boundary file was actually "
    "fetched or created earlier in this conversation — do not invent or "
    "guess a boundary file path. If no boundary file exists yet, geocode "
    "the region (bbox) via CDSE first and clip to that bbox instead, or "
    "obtain the boundary through a dedicated tool if one is available.\n"
    "\n"
    "- When delegating an imagery request to CDSE, state whether the output "
    "is for visual inspection (viewing, VLM) or for downstream quantitative "
    "use (GIS analysis server, NDVI/band math, zonal stats, more than 3 "
    "bands). CDSE should fetch a JPEG quicklook (≤3 bands, true-color/preset) "
    "for the former, and a GeoTIFF (as_geotiff=True) for the latter.\n"
    "- The same applies when delegating to GEE: state whether the output is "
    "for visual inspection (VLM) or downstream quantitative use. GEE should "
    "fetch a PNG quicklook (as_geotiff=False) for the former, and a raw "
    "GeoTIFF (as_geotiff=True, the default) for the latter — GEE's GeoTIFF "
    "output is not viewable by the VLM sub-agent.\n"
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
    logger.info("Session created: App='%s', User='%s', Session='%s'", app_name, user_id, session_id)
    return session

session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

runner = Runner(agent=root, # The agent we want to run
                app_name=APP_NAME,   # Associates runs with our app
                session_service=session_service # Uses our session manager
                )
logger.info("Runner created for agent '%s'.", runner.agent.name)

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





async def get_ans_tool_path(query: str) -> str:
    """
    Runs a query through the orchestrator in an isolated session and returns
    a JSON string with the final answer, the sequence of agents invoked, the
    leaf MCP tool calls (name + arguments) made to produce it, and total
    token usage across the run.
    """
    _orphan_guard.reset()

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
    input_tokens = 0
    output_tokens = 0

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.author and (not agents_called or agents_called[-1] != event.author):
            agents_called.append(event.author)

        if event.usage_metadata:
            input_tokens += event.usage_metadata.prompt_token_count or 0
            output_tokens += event.usage_metadata.candidates_token_count or 0

        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, 'function_call', None)
                if fc and not fc.name.endswith('agent'):
                    tools_called.append(
                        {"function_name": fc.name, "arguments": dict(fc.args)}
                    )

                fr = getattr(part, 'function_response', None)
                if fr is not None:
                    resp = fr.response
                    error_text = None
                    if isinstance(resp, dict):
                        if resp.get("error"):
                            error_text = str(resp["error"])
                        elif resp.get("isError"):
                            error_text = str(resp.get("content") or resp)
                    # Attach to the most recent still-unresolved call to this
                    # same tool name (matches by name since concurrent calls to
                    # the same sub-agent are disallowed by the system prompt).
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
        "agents_called": agents_called,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }, default=str)

import json
from collections import Counter

async def debug_agent_async(query: str, runner, user_id, session_id):
    """Verbose event dumper for diagnosing repeated/looping sub-agent or tool calls."""
    print(f"\n{'='*80}\n>>> USER QUERY: {query}\n{'='*80}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    call_log = []  # (author, name, args) for every function_call seen

    i = 0
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        print(f"\n--- event #{i} | author={event.author} | "
              f"final={event.is_final_response()} ---")

        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, 'text', None):
                    print(f"  [TEXT] {part.text}")
                fc = getattr(part, 'function_call', None)
                if fc:
                    args = dict(fc.args)
                    print(f"  [CALL] {fc.name}({json.dumps(args, indent=2, default=str)})")
                    call_log.append((event.author, fc.name, args))
                fr = getattr(part, 'function_response', None)
                if fr:
                    print(f"  [RESULT] {fr.name} -> "
                          f"{json.dumps(fr.response, indent=2, default=str)[:1500]}")

        # Sub-agent handoffs show up here, not as a normal function_call
        if event.actions:
            if getattr(event.actions, 'transfer_to_agent', None):
                print(f"  [TRANSFER] -> {event.actions.transfer_to_agent}")
            if getattr(event.actions, 'escalate', None):
                print(f"  [ESCALATE] {event.error_message}")

        i += 1

    print(f"\n{'='*80}\nCALL SUMMARY\n{'='*80}")
    counts = Counter((a, n) for a, n, _ in call_log)
    for (author, name), n in counts.items():
        print(f"  {author:>20} -> {name:<25} called {n}x")
        if n > 1:
            for a, nm, args in call_log:
                if (a, nm) == (author, name):
                    print(f"      args: {args}")

async def run_conversation():
    await debug_agent_async(

    "Search arXiv for papers on sea-level rise monitoring using satellite altimetry published since 2024 and return up to 15 results.",
    runner, USER_ID, SESSION_ID,
    )
if __name__ == "__main__":
    try:
        # query = input("Enter your question: ")
        ans = asyncio.run(run_conversation())
        print(ans)
    except Exception as e:
        print(f"An error occurred: {e}")