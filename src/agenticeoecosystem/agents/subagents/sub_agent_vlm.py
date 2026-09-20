# agenticeoecosystem/agents/subagents/vlm_agent.py

"""Generic vision-language sub-agent, with optional domain-specific analysis
knowledge activated per request (see `vlm_domains`).

By default this agent does plain, domain-agnostic visual analysis — the same
behavior it always had. An orchestrator that wants deeper domain-specific
criteria applied (e.g. wildfire visual cues) includes a `VLM_DOMAIN: <key>`
marker line in the handoff message it sends immediately before transferring
control here; this agent's instruction tells it to look for that marker and
apply the matching domain from `vlm_domains.DOMAINS`, or ignore it and stay
generic if the marker is absent or unrecognized.

This is intentionally the simple version (Option A: prompt convention over
inherited conversation history, no new tool or session state) rather than a
session-state-backed activation mechanism. That keeps this change scoped to
prompt content, but it has a real limitation worth knowing: because the
instruction string is assembled once at import time, every registered
domain's guidance text is always present in this agent's context — the
model chooses whether to apply it based on the marker, but a request that
never mentions any domain still has that text sitting in its instructions,
and within a single multi-turn session a domain marker from an earlier turn
could in principle be misapplied to a later, unrelated turn if the model
doesn't scope it correctly. If that proves unreliable in practice, the
documented follow-up is a session-state-backed activation: an explicit
`set_vlm_domain(domain)` tool the orchestrator calls before transferring,
paired with turning `instruction` here into a per-invocation callable
(ADK's InstructionProvider) that reads the active domain from session state
instead of relying on the model to self-scope a text marker. That gives a
structural guarantee instead of a behavioral convention.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from agenticeoecosystem.agents.subagents.vlm_domains import (
    extract_domain_marker,
    get_domain,
    render_domain_catalog,
)
from agenticeoecosystem.agents.subagents.vlm_init import vlm
from agenticeoecosystem.definitions import ROOT_DIR

logger = logging.getLogger(__name__)

_VLM_BASE_PROMPT = """You are a VLM reasoning agent.

You receive one or more file paths (a single image or a directory of images)
and a natural-language request from an orchestrator agent. Use the
`resolve_images` tool first to find out what images are available at each
given path, then use `load_images` to actually load them into context before
answering.

Rules:
- If the path is a directory, only the first 10 images (sorted by name) will
  ever be loaded — mention it if the directory contained more.
- If a path doesn't resolve to any valid image, say so clearly instead of
  guessing.
- Call `load_images` exactly ONCE per distinct path you were given — never
  call it twice with the same path. Each image is expensive context; a
  duplicate call wastes budget without adding information and can push the
  request over the model's context limit.
- If you were given multiple distinct paths to compare, load each one in
  turn (a different path every time) before answering — do not answer after
  loading only one of them.
- Always ground your answer in the images you were actually given.
"""


def _build_vlm_instruction() -> str:
    catalog = render_domain_catalog()
    if not catalog:
        return _VLM_BASE_PROMPT
    return (
        f"{_VLM_BASE_PROMPT}\n"
        "DOMAIN-SPECIFIC ANALYSIS:\n"
        "The message that triggered this handoff (the orchestrator's own "
        "text immediately before it transferred control to you) may contain "
        "a line of the form `VLM_DOMAIN: <key>`. If it does, and <key> "
        "matches one of the domains below, apply that domain's guidance in "
        "addition to the rules above. If no such line is present, or <key> "
        "doesn't match any domain below, ignore it and stick to the generic "
        "rules above — never guess a domain purely from image content, and "
        "never carry a domain over from an earlier, unrelated turn in this "
        "conversation.\n\n"
        f"{catalog}"
    )


def _log_domain_marker(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Observability only — logs which VLM_DOMAIN marker (if any) is in the
    latest turn. Does not alter the request; the instruction text above is
    static and already contains every registered domain's guidance."""
    contents = llm_request.contents or []
    last_text = ""
    if contents:
        last_text = "".join(
            part.text or "" for part in (contents[-1].parts or [])
        )
    key = extract_domain_marker(last_text)
    if key is None:
        return
    domain = get_domain(key)
    logger.info(
        "VLM handoff mentions VLM_DOMAIN='%s' (%s)",
        key, "recognized" if domain else "unrecognized, falling back to generic",
    )


_VLM_SYSTEM_PROMPT = _build_vlm_instruction()

toolset = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/vlm_tools/server.py"],
            ),
            timeout=300,
        )
    )
]


def _fetch_agent(toolset: list) -> Optional[Agent]:
    if not toolset:
        logger.warning("VLM agent not initialized due to missing toolset.")
        return None
    return Agent(
        model=vlm,
        name="vlm_agent",
        description="A helpful assistant for answering user questions related to reasoning over image data.",
        instruction=_VLM_SYSTEM_PROMPT,
        tools=toolset,
        before_model_callback=_log_domain_marker,
        # NOTE: this agent is registered as a native ADK sub-agent and reached
        # via the built-in transfer_to_agent (no message payload) rather than
        # as an agent-tool with an explicit `request` argument — so it relies
        # entirely on inherited session history to learn what image paths it's
        # meant to work on. include_contents='none' silently breaks that (it
        # gets transferred control with no idea what to do). Do NOT set it
        # here; the per-image token cost is instead controlled in load_images
        # (vlm_tools/server.py), which downscales before base64-encoding.
    )


vlm_agent = _fetch_agent(toolset)