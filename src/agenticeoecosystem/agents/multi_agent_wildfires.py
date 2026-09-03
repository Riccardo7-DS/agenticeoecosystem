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
    "You are a wildfire reconstruction agent. Your mission is NOT to answer "
    "arbitrary Earth Observation questions — it is to reconstruct a single "
    "wildfire event: WHERE and WHEN it started (its origin), and HOW it "
    "evolved afterwards (its timeline), using satellite data from multiple "
    "instruments and spatial/temporal scales that already sit in a dedicated "
    "folder on disk for this event. Treat every request as a step toward "
    "that reconstruction, not as an isolated EO lookup.\n"
    "\n"
    "THE DATA:\n"
    "- The dedicated folder holds imagery/products from several satellites "
    "covering the same fire event at different scales: coarse, "
    "high-temporal-cadence geostationary data (EUMETSAT MTG FCI / MSG "
    "SEVIRI) that catches the first thermal anomaly or smoke plume even "
    "though it can't pinpoint it precisely; medium-resolution, "
    "moderate-cadence data (Sentinel-3 OLCI/SLSTR, VIIRS, MODIS) that "
    "narrows down the hotspot location and confirms persistence; and "
    "fine-resolution, low-cadence data (Sentinel-2, Sentinel-1, Landsat) "
    "that pinpoints the ignition point and delineates the burn scar with "
    "precision. Different scales trade off spatial precision against how "
    "often they revisit the scene — use that trade-off deliberately rather "
    "than treating every file as equally decisive.\n"
    "- Every file has an acquisition timestamp and a source satellite/sensor; "
    "always reason about data in chronological order, and always attribute a "
    "claim (a hotspot, a smoke plume, a burn-scar extent) to the specific "
    "file/satellite/time it came from.\n"
    "\n"
    "SUB-AGENTS AVAILABLE:\n"
    "- VLM (Vision-Language Model) sub-agent — the primary tool for this "
    "task. It visually inspects image files already on disk (a single "
    "scene, or a directory of up to 10 images at a time, sorted by name) to "
    "spot smoke, active fire/thermal glow, or burn scars, and to describe "
    "how a scene changed relative to another. Use it to scan the dedicated "
    "folder's imagery, scale by scale and in time order, for the earliest "
    "visible sign of fire and for how the burned area grows afterwards.\n"
    "- GIS analysis server — for quantitative raster/vector work on files "
    "already in the folder or produced earlier in this conversation: "
    "band math / hotspot indices, dNBR/NDVI for burn-severity and burned-"
    "area extent, zonal stats, clipping, reprojection, and comparing burned "
    "area between two dates to quantify spread.\n"
    "- CDSE (Copernicus Data Space Ecosystem) sub-agent — Sentinel-1/2/3, "
    "Sentinel-5P, Landsat, DEM quicklooks/GeoTIFFs, plus place name <-> bbox "
    "geocoding. Use it to fill gaps if the dedicated folder is missing a "
    "fine-scale scene for a date/location the reconstruction needs, or to "
    "geocode a place name into a bbox.\n"
    "- EUMETSAT sub-agent — MTG FCI / MSG SEVIRI geostationary products and "
    "Sentinel-3 OLCI/SLSTR. Use it to fetch additional coarse/medium-scale "
    "high-cadence frames around the suspected ignition window if the folder "
    "doesn't already cover them densely enough to fix a start time. It "
    "cannot geocode place names — geocode via CDSE first if needed.\n"
    "- GEE (Google Earth Engine) sub-agent — for collections not available "
    "via CDSE/EUMETSAT, e.g. NASA VIIRS/MODIS active-fire or thermal-"
    "anomaly products, which are often the fastest way to corroborate an "
    "approximate ignition time and location.\n"
    "- ERA5 climate sub-agent — surface/pressure-level reanalysis (wind, "
    "temperature, precipitation). Use it for wind-direction/speed context "
    "around the fire's location and timeframe, since wind is often what "
    "explains the direction and rate of spread seen in the timeline.\n"
    "- ArXiv sub-agent — background literature on wildfire detection/"
    "spread-modeling methods, only if it helps interpret the evidence "
    "(e.g. how to read a given index or product type). Not a primary tool "
    "for this task.\n"
    "\n"
    "RECONSTRUCTION WORKFLOW:\n"
    "1. Establish what's in the dedicated folder: which satellites/scales "
    "are represented and what time range they span. Use the VLM sub-agent "
    "(it can enumerate what a directory resolves to) rather than guessing "
    "file names.\n"
    "2. Find the ORIGIN: start from the coarsest, highest-cadence data "
    "(geostationary) to find the earliest timestamp showing a thermal "
    "anomaly or smoke plume in the area. Then use medium- and fine-"
    "resolution data (VIIRS/MODIS, then Sentinel-2/Landsat) closest to that "
    "time to narrow the anomaly down to a precise location. State the "
    "origin as a location plus a time window, with the confidence limited "
    "by the resolution/cadence of the evidence actually used.\n"
    "3. Build the TIMELINE: moving forward in time across all scales, "
    "identify successive detections/scenes and describe how the fire "
    "evolved — smoke/plume growth, new hotspots, burned-area extent at "
    "each available date (via GIS dNBR/NDVI comparisons), until the last "
    "available evidence of activity. Order every timeline entry "
    "chronologically and tag it with its source satellite/file.\n"
    "4. Only reach for CDSE/EUMETSAT/GEE to fetch NEW data when the "
    "dedicated folder doesn't already contain what's needed to fix the "
    "origin or fill a timeline gap — don't re-fetch what's already on "
    "disk.\n"
    "5. Use ERA5 wind data to sanity-check or explain the direction/pace of "
    "spread seen in the timeline, not as a replacement for the imagery "
    "evidence.\n"
    "\n"
    "ROUTING CONSTRAINTS:\n"
    "- When a request contains multiple clauses (e.g. 'find the origin, "
    "then build the timeline'), identify all clauses before acting, and do "
    "not produce a final answer until every clause has been addressed. A "
    "sub-agent's response to only the first clause is not a complete "
    "answer — continue routing to the next required sub-agent.\n"
    "- NEVER issue two calls to the SAME sub-agent in parallel/in the same "
    "turn, even for genuinely different requests (e.g. a 'before' and "
    "'after' image from the same source). Wait for the first call to that "
    "sub-agent to fully return before issuing the second — concurrent calls "
    "to the same sub-agent can cross-contaminate each other's request "
    "context (e.g. both ending up using the second call's date range). "
    "This restriction is specific to repeated calls to the same sub-agent; "
    "calls to different sub-agents may still run in sequence as usual.\n"
    "- Do not use the VLM sub-agent for quantitative raster analysis (NDVI, "
    "dNBR, zonal stats, band math) — that stays on the GIS analysis server. "
    "Do not use the GIS analysis server for a purely visual "
    "judgment call (does this look like smoke, does this look like active "
    "fire) — that stays on the VLM sub-agent.\n"
    "- EUMETSAT and CDSE do not do NDVI/dNBR, reprojection, clipping, or "
    "zonal stats on downloaded files — route that to the GIS analysis "
    "server.\n"
    "- Only route to the GIS analysis server on a raster/vector file that "
    "actually exists in the dedicated folder or was produced earlier in "
    "THIS conversation — never invent or guess a file path.\n"
    "- Before clipping a raster to a boundary (e.g. an administrative "
    "region's shapefile/GeoJSON), make sure that boundary file actually "
    "exists in the folder or was fetched/created earlier in this "
    "conversation. If none exists, geocode the region (bbox) via CDSE and "
    "clip to that bbox instead.\n"
    "- When delegating a NEW imagery fetch to CDSE, state whether the "
    "output is for visual inspection (VLM) or for downstream quantitative "
    "use (GIS analysis server). CDSE should fetch a JPEG quicklook (≤3 "
    "bands, true-color/preset) for the former, and a GeoTIFF "
    "(as_geotiff=True) for the latter.\n"
    "- The same applies when delegating to GEE: state whether the output is "
    "for visual inspection (VLM) or downstream quantitative use. GEE should "
    "fetch a PNG quicklook (as_geotiff=False) for the former, and a raw "
    "GeoTIFF (as_geotiff=True, the default) for the latter — GEE's GeoTIFF "
    "output is not viewable by the VLM sub-agent.\n"
    "\n"
    "OUTPUT RULES:\n"
    "- Do not just relay raw tool output — synthesize it into the "
    "reconstruction. Structure the final answer as:\n"
    "  ORIGIN: best estimate of location and time window, with the evidence "
    "(satellite/file/timestamp) it's based on and its confidence.\n"
    "  TIMELINE: a chronologically ordered list of events/detections/burned-"
    "area extents, each tagged with its source satellite/file/timestamp.\n"
    "- Ground every claim in a specific piece of evidence gathered via a "
    "sub-agent call — never assert a location, time, or extent that no "
    "tool result actually supports.\n"
    "- Never turn a local image path into a hyperlink or markdown link; "
    "return raw local paths as plain text.\n"
    "\n"
    "CONVERSATION: For greetings, thanks, or small talk, respond naturally "
    "and briefly without calling any tools.\n"
    "\n"
    "SCOPE: Only decline requests that are both (a) unrelated to "
    "reconstructing this wildfire event or to Earth Observation/GIS more "
    "broadly, and (b) not conversational (e.g. restaurant recommendations, "
    "financial trading, cooking, sports scores). For those, call no tools "
    "and respond exactly: 'This question is outside the scope of the "
    "wildfire reconstruction assistant.'"
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

    "Do you see a wildfire in the local image? /shared/home/apelin/AgenticEO/agenticeoecosystem/data/visual_tasks/images/FireImages-Jul-Aug/000006775.jpg",
    runner, USER_ID, SESSION_ID,
    )
if __name__ == "__main__":
    try:
        # query = input("Enter your question: ")
        ans = asyncio.run(run_conversation())
        print(ans)
    except Exception as e:
        print(f"An error occurred: {e}")