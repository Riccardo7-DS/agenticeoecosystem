"""Code shared by multi_agent.py and multi_agent_research.py.

Moved here, unchanged, so the research orchestrator can reuse it without importing
multi_agent.py, which builds its own root agent over the same subagent objects
(an ADK agent can have only one parent).
"""
import logging
import re

from agenticeoecosystem.agents.subagents.eve_patch import _PLACEHOLDER_ASSISTANT_CONTENT

logger = logging.getLogger(__name__)

# ============================ Supervisor prompt ============================
SUPERVISOR_SYSTEM_PROMPT = (
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

_head, _marker, _tail = SUPERVISOR_SYSTEM_PROMPT.partition("OUTPUT RULES:")
if not _marker:
    raise RuntimeError("'OUTPUT RULES:' not found in SUPERVISOR_SYSTEM_PROMPT; update the split.")
SUPERVISOR_ROUTING = _head                  # which sub-agent to use, and when
SUPERVISOR_OUTPUT_RULES = _marker + _tail   # how to answer, conversation, scope


# ============================ Tool-error recovery ============================

def on_tool_error(tool, args, tool_context, error):
    """Turn a tool-execution failure into a normal tool result instead of
    letting it propagate and abort the turn (which otherwise leaves that
    tool call's id permanently unresolved in history — e.g. when the model
    hallucinates a call to a tool that isn't registered, such as
    transfer_to_agent, which this orchestrator never exposes).

    OrphanedToolCallStormError is deliberately excluded from this recovery:
    it's raised by orphaned_tool_call_guard specifically to abort a stuck
    task by propagating all the way to pipeline_runner.py's per-task
    try/except. Swallowing it here (as every other tool error is swallowed)
    defeated that guard entirely — the orchestrator would just catch the
    "error", hand it back to the model as a normal tool result, and keep
    looping, since AgentTool re-invokes the sub-agent as a plain tool call
    that flows through this same callback. Returning None here lets ADK's
    functions.py fall through to `raise tool_error`, so the task is
    correctly recorded as unsolved instead of retry-storming forever.
    """
    from agenticeoecosystem.agents.subagents.orphaned_tool_call_guard import (
        OrphanedToolCallStormError,
    )
    if isinstance(error, OrphanedToolCallStormError):
        logger.error("Tool '%s' hit storm guard, aborting task: %s", getattr(tool, "name", "?"), error)
        return None

    logger.warning("Tool '%s' failed: %s", getattr(tool, "name", "?"), error)
    return {"error": f"Tool '{getattr(tool, 'name', '?')}' failed: {error}"}


# ============================ Response text cleaning ============================
# Some models (observed on gemma4-31b served via vLLM — see the "gemma4-31b"
# entry in utils/registry.py, which has no reasoning_parser configured
# because no vLLM-registered parser matches this model's channel-tag shape)
# leak a malformed reasoning/"thought"-channel marker as a literal prefix on
# the final assistant text instead of it being split out server-side, e.g.:
#   "<|channel>thought\n<channel|>The available themes are: ..."
# vLLM's real harmony-style channel format is "<|channel|>NAME<|message|>",
# so this is already a mangled/partial token sequence that leaked through as
# plain text rather than a normal special token — no --reasoning-parser flag
# reliably strips it. Strip it defensively here instead, so it never lands in
# a recorded final_answer.
_LEAKED_CHANNEL_PREFIX_RE = re.compile(
    r"^\s*<\|?channel\|?>\s*[\w-]*\s*\n?\s*<\s*channel\|?>\s*",
    re.IGNORECASE,
)


def strip_leaked_channel_marker(text: str) -> str:
    """Remove a leaked reasoning-channel marker prefix, if present."""
    if not text:
        return text
    return _LEAKED_CHANNEL_PREFIX_RE.sub("", text, count=1)


# DeepSeek-R1-family models (e.g. deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) wrap
# reasoning in literal <think>...</think> tags. The intended fix is hosting
# with --reasoning-parser deepseek_r1 so vLLM splits this out server-side
# into a separate reasoning_content field (see
# vllm.reasoning.deepseek_r1_reasoning_parser). If that flag is missing, or
# only a partial tag leaks through anyway, defend the same way as the
# channel-marker case above. Unlike that case, a <think> block can run for
# many paragraphs and may never close at all if the model hits max_tokens
# mid-thought — surfacing raw, unclosed chain-of-thought as a "final answer"
# would be worse than reporting no answer, so an unclosed block returns None
# instead of the leaked text.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>", re.IGNORECASE)


def strip_leaked_think_block(text: str) -> str | None:
    """Remove a leaked <think>...</think> reasoning block, if present.
    Returns None if the block is unclosed (truncated mid-thought) rather
    than surfacing partial chain-of-thought as if it were a real answer.
    """
    if not text:
        return text
    stripped = _THINK_BLOCK_RE.sub("", text, count=1)
    if stripped != text:
        return stripped
    if _UNCLOSED_THINK_RE.match(text):
        return None
    return text


def clean_text(content) -> str | None:
    """
    Pull plain text out of a Content (e.g. a final response event's), if any.
    Guards against parts that are function_call / function_response
    (which don't have a usable `.text`).

    Also guards against eve_patch's synthetic role-alternation filler
    ("(continuing)") being mistaken for a real final answer: ADK replays a
    finished sub-agent's transcript back into the parent's history as
    synthetic events, and that replay can surface our own injected
    placeholder (inserted purely to satisfy OpenAI's tool->assistant role
    ordering) as if it were a genuine is_final_response() event — silently
    discarding the model's actual answer. Observed affecting ~5-15% of
    tasks across every "multi"-pipeline candidate run.

    Also skips reasoning/"thought" parts. When a server-side reasoning
    parser (e.g. vLLM's --reasoning-parser deepseek_r1) correctly splits a
    model's chain-of-thought into a separate reasoning_content field, ADK's
    LiteLlm wrapper (_convert_reasoning_value_to_parts in
    google/adk/models/lite_llm.py) converts it into its own
    types.Part(text=..., thought=True) — prepended before the real content
    part in the *same* event.content.parts list. Without this check we'd
    return that reasoning text as the "final answer" (it has a normal
    .text attribute), before ever reaching the actual content part —
    exactly the kind of chain-of-thought leak that would bias an evaluator
    LLM's judgment, just via a different path than a raw <think> tag leak.
    """
    if not (content and getattr(content, "parts", None)):
        return None
    for part in content.parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text and text.strip() == _PLACEHOLDER_ASSISTANT_CONTENT:
            continue
        if text:
            cleaned = strip_leaked_think_block(strip_leaked_channel_marker(text))
            if cleaned is not None:
                return cleaned
    return None


def extract_final_text(event) -> str | None:
    """Plain text of a final-response event (see clean_text)."""
    return clean_text(event.content)