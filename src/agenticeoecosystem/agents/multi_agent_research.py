"""Research orchestrator: the same sub-agents as multi_agent.py, plus the hypothesis agent.

Hypotheses stay internal: they shape which questions get asked, never the answer text.
Do NOT import this module in the same process as multi_agent.py (see orchestrator_common.py).
"""
# Same patches as multi_agent.py, in the same order.
from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
apply_eve_patch()
from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as apply_mcp_image_patch
apply_mcp_image_patch()
from agenticeoecosystem.agents.subagents.strict_tool_patch import apply as apply_strict_tool_patch
apply_strict_tool_patch()
from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard as _orphan_guard
_orphan_guard.apply()
from agenticeoecosystem.agents.subagents import arxiv_rate_limit_retry as _arxiv_retry
_arxiv_retry.apply()

import json
import logging
import uuid

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agenticeoecosystem.agents import helper
from agenticeoecosystem.agents.helper.ledger import get_hypothesis_summary
from agenticeoecosystem.agents.orchestrator_common import (
    SUPERVISOR_OUTPUT_RULES, SUPERVISOR_ROUTING, extract_final_text, on_tool_error,
)
from agenticeoecosystem.agents.subagents import ALL_SUB_AGENTS, llm
from agenticeoecosystem.agents.subagents.hypothesis_agent import build_hypothesis_agent

load_dotenv()
logger = logging.getLogger(__name__)

helper.configure("research")
helper.attach_subagents(ALL_SUB_AGENTS)
hypothesis_agent = build_hypothesis_agent([a.name for a in ALL_SUB_AGENTS])

# Calls by these never enter tools_called, so a judge never scores them as tool use.
HELPER_AUTHORS = {hypothesis_agent.name}
HELPER_TOOLS = {"get_hypothesis_summary"}

# DECISION: how the coordinator uses hypotheses. Edit freely; avoid {braces} (ADK state syntax).
_INVESTIGATION = """
INVESTIGATION:
- For research questions that need evidence from several sources, work in rounds.
  After every 2-3 sub-agent answers that report new results, call the hypothesis sub-agent.
- Use its NEXT QUESTIONS to plan the next round, preferring questions where the hypotheses
  predict different outcomes. Stop when it has no NEXT QUESTIONS, or after 3 rounds.
- Call get_hypothesis_summary() to see the current hypotheses without changing them.
- For plain data requests (fetch, compute, look up), do not call the hypothesis sub-agent.

"""
_HYPOTHESES_STAY_INTERNAL = """

HYPOTHESES ARE INTERNAL: the hypothesis sub-agent's output and get_hypothesis_summary() are
for planning only. Never relay them, and never mention hypotheses or the ledger, in your
answer to the user. The OUTPUT RULES above apply to the other sub-agents' results."""

_RESEARCH_PROMPT = SUPERVISOR_ROUTING + _INVESTIGATION + SUPERVISOR_OUTPUT_RULES + _HYPOTHESES_STAY_INTERNAL

root = Agent(
    model=llm,
    name="research_orchestrator_agent",
    sub_agents=ALL_SUB_AGENTS + [hypothesis_agent],
    description="An orchestrator that investigates research questions with its sub-agents "
                "and keeps an internal hypothesis ledger.",
    instruction=_RESEARCH_PROMPT,
    tools=[get_hypothesis_summary],
    **helper.root_callbacks(on_tool_error),
)

APP_NAME = "EVE-AGENT-RESEARCH"
USER_ID = "user_1"
session_service = InMemorySessionService()
runner = Runner(agent=root, app_name=APP_NAME, session_service=session_service)


def _attach_result(calls: list[dict], fr) -> None:
    """Same matching as multi_agent.get_ans_tool_path."""
    resp, error_text = fr.response, None
    if isinstance(resp, dict):
        if resp.get("error"):
            error_text = str(resp["error"])
        elif resp.get("isError"):
            error_text = str(resp.get("content") or resp)
    for entry in reversed(calls):
        if entry["function_name"] == fr.name and "tool_result" not in entry:
            entry["tool_result"] = resp
            if error_text:
                entry["error"] = error_text
            break


async def get_ans_tool_path(query: str) -> str:
    """Same output as multi_agent.get_ans_tool_path, plus 'helper_calls' (kept out of tools_called)."""
    _orphan_guard.reset()
    session_id = f"session_{uuid.uuid4().hex}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_answer = "Agent did not produce a final response."
    agents_called, tools_called, helper_calls = [], [], []
    input_tokens = output_tokens = 0

    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if event.author and (not agents_called or agents_called[-1] != event.author):
            agents_called.append(event.author)
        if event.usage_metadata:
            input_tokens += event.usage_metadata.prompt_token_count or 0
            output_tokens += event.usage_metadata.candidates_token_count or 0

        from_helper = event.author in HELPER_AUTHORS
        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc and not fc.name.endswith("agent"):
                    target = helper_calls if (from_helper or fc.name in HELPER_TOOLS) else tools_called
                    target.append({"function_name": fc.name, "arguments": dict(fc.args)})
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    _attach_result(helper_calls if (from_helper or fr.name in HELPER_TOOLS) else tools_called, fr)

        if event.is_final_response() and not from_helper:
            text = extract_final_text(event)
            if text is not None:
                final_answer = text
            elif event.actions and event.actions.escalate:
                final_answer = f"Agent escalated: {event.error_message or 'No specific message.'}"

    return json.dumps({
        "final_answer": final_answer,
        "ended_in_error": bool(tools_called) and "error" in tools_called[-1],
        "tools_called": tools_called,
        "agents_called": agents_called,
        "helper_calls": helper_calls,
        "input_tokens": input_tokens,     # includes the hypothesis agent: it is part of this system
        "output_tokens": output_tokens,
    }, default=str)