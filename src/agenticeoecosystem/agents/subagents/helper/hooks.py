"""ADK callbacks: the only part of the helper that ADK calls directly.
Every callback returns None, so none of them changes what the agents do."""
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agenticeoecosystem.agents.helper import config
from agenticeoecosystem.agents.helper.ledger import ledger
from agenticeoecosystem.agents.helper.reviewer import evaluate_and_dispatch, spawn
from agenticeoecosystem.agents.helper.trace import Interaction, TraceStep, to_text
from agenticeoecosystem.agents.orchestrator_common import clean_text

logger = logging.getLogger(__name__)

TASK_TOOL_PREFIX = "request_task_"   # ASSUMPTION: verify with the probe


class RunState:
    """Everything about the current run (one task). Replaced when a new invocation starts."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.run_dir: Optional[Path] = config.run_dir(run_id) if run_id else None
        self.research_question = ""
        self.pending_questions: dict[str, str] = {}
        self.open: dict[str, list[Interaction]] = defaultdict(list)


run = RunState()


def _agent_from_tool(tool) -> Optional[str]:
    return tool.name[len(TASK_TOOL_PREFIX):] if tool.name.startswith(TASK_TOOL_PREFIX) else None


def _append_index(r: RunState) -> None:
    """One line per run, to match a task in results_<label>.json (by its text) to its reports."""
    idx = config.output_root() / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("a") as f:
        f.write(json.dumps({"run_id": r.run_id, "mode": config.MODE, "started_at": time.time(),
                            "question": r.research_question}) + "\n")

# ============================== ROOT ==============================

def root_before_agent(callback_context):
    global run
    if callback_context.invocation_id != run.run_id:
        run = RunState(callback_context.invocation_id)
        run.research_question = clean_text(callback_context.user_content) or ""
        if config.feeds_hypotheses():
            ledger.reset(run.run_id, run.run_dir)
        _append_index(run)
    return None


def root_before_tool(tool, args, tool_context):
    agent = _agent_from_tool(tool)
    if agent in config.EVALUATED_AGENTS:
        # ASSUMPTION: one string arg is the question; otherwise keep all args.
        strings = [v for v in args.values() if isinstance(v, str)]
        run.pending_questions[agent] = strings[0] if len(strings) == 1 else json.dumps(args, default=str)
    return None


def root_after_tool(tool, args, tool_context, tool_response):
    agent = _agent_from_tool(tool)
    if agent not in config.EVALUATED_AGENTS:
        return None
    stack = run.open.get(agent)
    if not stack:
        logger.warning("No open interaction for %s: did its callbacks fire?", agent)
        return None
    it = stack.pop()
    # Prefer what actually crossed the boundary to the coordinator.
    it.answer = to_text(tool_response, limit=20000) if tool_response else it.last_model_text
    spawn(evaluate_and_dispatch(it, run.research_question, run.run_id, run.run_dir))
    return None


def root_on_tool_error(tool, args, tool_context, error):
    agent = _agent_from_tool(tool)
    if agent and run.open.get(agent):
        run.open[agent].pop()   # DECISION: failed interactions are discarded, not reviewed
    return None

# ============================ SUBAGENTS ============================

def sub_before_agent(callback_context):
    name = callback_context.agent_name
    q = run.pending_questions.pop(name, None)
    source = "task_tool_args"
    if q is None:
        q = clean_text(callback_context.user_content) or ""
        source = "user_content_fallback"   # may be the USER prompt, not the coordinator's question
    run.open[name].append(Interaction(agent_name=name, question=q, question_source=source))
    return None


def sub_after_tool(tool, args, tool_context, tool_response):
    stack = run.open.get(tool_context.agent_name)
    if stack:
        it = stack[-1]
        it.steps.append(TraceStep(len(it.steps), tool.name, args, to_text(tool_response)))
    return None


def sub_after_model(callback_context, llm_response):
    if getattr(llm_response, "partial", False):
        return None
    stack = run.open.get(callback_context.agent_name)
    text = clean_text(llm_response.content)   # skips thoughts, placeholders, leaked reasoning
    if stack and text and text.strip():
        stack[-1].last_model_text = text
    return None