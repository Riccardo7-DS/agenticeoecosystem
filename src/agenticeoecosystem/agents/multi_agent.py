from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
from agenticeoecosystem.agents.subagents.eve_patch import _PLACEHOLDER_ASSISTANT_CONTENT
apply_eve_patch()

from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as apply_mcp_image_patch
apply_mcp_image_patch()

from agenticeoecosystem.agents.subagents.strict_tool_patch import apply as apply_strict_tool_patch
apply_strict_tool_patch()

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

from agenticeoecosystem.agents import helper
from agenticeoecosystem.agents.orchestrator_common import (
    SUPERVISOR_SYSTEM_PROMPT as _SUPERVISOR_SYSTEM_PROMPT,
    extract_final_text as _extract_final_text,
    on_tool_error as _on_tool_error,
    strip_leaked_channel_marker as _strip_leaked_channel_marker,
    strip_leaked_think_block as _strip_leaked_think_block,
)

# AGENTICEO_HELPER: off (default) or review. See agents/helper/config.py.
_HELPER_MODE = os.getenv("AGENTICEO_HELPER", "off")
if _HELPER_MODE == "research":
    raise ValueError("AGENTICEO_HELPER=research is only for multi_agent_research.py")
helper.configure(_HELPER_MODE)
helper.attach_subagents(ALL_SUB_AGENTS)   # must run before root is built

root = Agent(
    model=llm,
    name="orchestrator_agent",
    sub_agents=ALL_SUB_AGENTS,
    description="An orchestrator agent that routes user queries to its sub-agents based on the query content.",
    instruction=_SUPERVISOR_SYSTEM_PROMPT,
    **helper.root_callbacks(_on_tool_error),
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
import re
import uuid

from google.genai import types  # adjust import to match your actual SDK path

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
                        if entry["function_name"] == fr.name and "tool_result" not in entry:
                            entry["tool_result"] = resp
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

    "Can you get the most recent imagery related to the forest fires near bordeaux, france",
    runner, USER_ID, SESSION_ID,
    )
if __name__ == "__main__":
    try:
        # query = input("Enter your question: ")
        ans = asyncio.run(run_conversation())
        print(ans)
    except Exception as e:
        print(f"An error occurred: {e}")
