"""
pipeline_runner.py — Runs your agentic pipeline on a TaskSet and records results.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

from tqdm import tqdm

from agenticeoecosystem.definitions import GEOBENCH_RESULTS
from .dataclasses import Solution, Step, TaskSet

logger = logging.getLogger(__name__)

_TOOL_ERROR_PREFIX = "TOOL ERROR:"


def ended_in_error(solution: Optional[Solution]) -> bool:
    """
    True if a solution's last recorded step is a tool call that errored out —
    i.e. the trace was cut short by a failure rather than reaching a real
    answer. Distinguishes a "silently degraded" solve (e.g. a timed-out or
    404'd MCP call left as the final step) from a genuine solve, since
    `generated_solution is not None` alone can't tell them apart.
    """
    if not solution or not solution.steps:
        return False
    comment = solution.steps[-1].comment
    return bool(comment) and comment.startswith(_TOOL_ERROR_PREFIX)


def run_pipeline(
    task_text: str,
    pipeline: str = "multi",
    task_timeout: Optional[float] = None,
) -> tuple[Solution, int, int]:
    """
    Run the agent pipeline on a single task and return the MCP-level tool steps.

    Args:
        task_text: The natural-language task prompt.
        pipeline : "multi" (google-adk orchestrator + native sub-agents — the
                   current multi-agent system, see agents/multi_agent.py),
                   "single" (one LangGraph agent, all tools), or
                   "multi_legacy" (older LangGraph supervisor + sub-agents in
                   client.py — superseded by "multi"; missing cdse, eumetsat,
                   and vlm sub-agents).

    Returns:
        solution     : Solution with one Step per MCP tool call made.
        input_tokens : Total input tokens across all agents.
        output_tokens: Total output tokens across all agents.
    """
    import nest_asyncio
    nest_asyncio.apply()

    if pipeline == "multi":
        import json as _json
        import agenticeoecosystem.agents.multi_agent as _adk

        loop = asyncio.get_event_loop()
        coro = _adk.get_ans_tool_path(task_text)
        if task_timeout:
            coro = asyncio.wait_for(coro, timeout=task_timeout)
        raw = loop.run_until_complete(coro)
        data = _json.loads(raw)
        steps = [
            Step(
                function_name=t["function_name"],
                arguments=t.get("arguments", {}),
                comment=(f"{_TOOL_ERROR_PREFIX} {t['error']}" if t.get("error") else None),
            )
            for t in data.get("tools_called", [])
        ]
        return (
            Solution(steps=steps, final_answer=data.get("final_answer")),
            data.get("input_tokens", 0),
            data.get("output_tokens", 0),
        )

    if pipeline == "single":
        import agenticeoecosystem.client.client_single_agent as _module
        _agent = _module.agent
    else:  # "multi_legacy"
        import agenticeoecosystem.client.client as _module
        _agent = _module.supervisor_agent

    _module.reset_step_log()

    async def _run() -> tuple[int, int]:
        async for chunk in _agent.astream(
            {"messages": [{"role": "user", "content": task_text}]},
            config={"recursion_limit": 50},
        ):
            for update in chunk.values():
                messages = update.get("messages", [])
                if pipeline == "single":
                    # Single agent calls MCP tools directly, so its own top-level
                    # messages carry the real tool_calls — log them here, mirroring
                    # how client.py's sub-agent wrappers log steps for "multi".
                    _module._extract_steps_from_messages(messages)
                else:
                    # For "multi_legacy", each sub-agent wrapper already logs its
                    # own MCP-level steps via _extract_steps_from_messages(); only
                    # the supervisor's token usage needs picking up here.
                    for msg in messages:
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            _module._token_counts["input"] += msg.usage_metadata.get("input_tokens", 0)
                            _module._token_counts["output"] += msg.usage_metadata.get("output_tokens", 0)
        return _module._token_counts["input"], _module._token_counts["output"]

    loop = asyncio.get_event_loop()
    in_tok, out_tok = loop.run_until_complete(_run())

    steps = [
        Step(function_name=s["function_name"], arguments=s["arguments"])
        for s in _module._step_log
    ]
    return Solution(steps=steps), in_tok, out_tok


# ---------------------------------------------------------------------------
# Batch runner — mirrors GeoBenchX generate_solutions()
# ---------------------------------------------------------------------------

def generate_solutions(
    tasks: TaskSet,
    model: str,
    temperature: float,
    model_provider: str = "ollama",
    output_filename: Optional[str] = None,
    output_folder: Optional[str] = None,
    skip_solved: bool = True,
    wait_between_tasks: float = 0.0,
    pipeline: str = "multi",
    agent_models: Optional[dict] = None,
    task_timeout: Optional[float] = None,
) -> tuple[TaskSet, int, int]:
    """
    Iterate over all tasks in `tasks`, call `run_pipeline`, and store results.

    Args:
        tasks            : TaskSet to process.
        model            : Model identifier (e.g. "nemotron-cascade-2:latest").
        temperature      : Sampling temperature.
        model_provider   : LangChain provider string (e.g. "ollama", "anthropic") or "eve".
        output_filename  : If set, save intermediate results after every task.
        output_folder    : Directory for intermediate saves (default: GEOBENCH_RESULTS).
        skip_solved      : Skip tasks that already have a generated_solution.
        wait_between_tasks: Seconds to wait between API calls (rate-limit guard).
        pipeline         : "multi" (google-adk orchestrator, the current
                           multi-agent system — all sub-agents share whatever
                           model `model` names; model_provider "eve" routes to
                           the EVE gateway, "vllm" routes to a local vLLM
                           server, any other value falls back to "eve"),
                           "single" (one LangGraph agent, all tools), or
                           "multi_legacy" (older LangGraph supervisor +
                           sub-agents in client.py — superseded by "multi").
        agent_models     : Per-agent model overrides for pipeline="multi_legacy",
                           e.g. {"gis": ("mistral-small-24b", "vllm")}. Ignored
                           for pipeline="single" or pipeline="multi".
        task_timeout     : Max wall-clock seconds for a single task's run_pipeline()
                           call. A timed-out task raises asyncio.TimeoutError, which
                           is caught below like any other per-task error — the loop
                           moves on to the next task instead of hanging indefinitely.

    Returns:
        tasks        : Same TaskSet with generated_solution fields populated.
        total_input  : Sum of input tokens across all tasks.
        total_output : Sum of output tokens across all tasks.
    """
    _save_folder = output_folder or GEOBENCH_RESULTS
    import nest_asyncio
    nest_asyncio.apply()

    logger.info(
        "Rebuilding agents — pipeline=%s  model=%s  provider=%s  temperature=%s",
        pipeline, model, model_provider, temperature,
    )

    if pipeline == "single":
        import agenticeoecosystem.client.client_single_agent as _single
        if agent_models:
            logger.warning("agent_models is ignored when pipeline='single'")
        _single.rebuild_agent(model, provider=model_provider, temperature=temperature)
    elif pipeline == "multi":
        if agent_models:
            logger.warning("agent_models is ignored when pipeline='multi' (ADK orchestrator)")
        if model_provider not in ("eve", "vllm"):
            logger.warning(
                "pipeline='multi' only supports provider='eve' (EVE gateway) or "
                "'vllm' (local server) — ignoring model_provider=%s, falling back "
                "to 'eve'", model_provider,
            )
            model_provider = "eve"
        if "agenticeoecosystem.agents.multi_agent" in sys.modules:
            logger.warning(
                "agenticeoecosystem.agents.multi_agent was already imported earlier "
                "in this process with a different MULTI_AGENT_MODEL/PROVIDER — "
                "model=%s provider=%s will have no effect. Each model needs its "
                "own fresh process.", model, model_provider,
            )
        os.environ["MULTI_AGENT_MODEL"] = model
        os.environ["MULTI_AGENT_PROVIDER"] = model_provider
        import agenticeoecosystem.agents.multi_agent  # noqa: F401 (import triggers setup)
    else:  # "multi_legacy"
        import agenticeoecosystem.client.client as _client
        _client.rebuild_agents(model, provider=model_provider, temperature=temperature, agent_models=agent_models)

    tasks.metadata["model"] = model
    tasks.metadata["temperature"] = temperature
    tasks.metadata["pipeline"] = pipeline

    n_total = len(tasks)
    n_skip = sum(1 for t in tasks if t.generated_solution is not None and skip_solved)
    logger.info("Generation phase — %d task(s) total, %d will be skipped (already solved)", n_total, n_skip)

    total_input = 0
    total_output = 0

    for i, task in enumerate(tqdm(tasks, desc="Generating solutions"), 1):
        logger.info("[%d/%d] Task %s: %s", i, n_total, task.task_ID, task.task_text[:80])

        if task.generated_solution is not None and skip_solved:
            logger.info("  → Skipped (already solved, %d steps)", len(task.generated_solution))
            continue

        t0 = time.time()
        try:
            solution, inp_tok, out_tok = run_pipeline(task.task_text, pipeline=pipeline, task_timeout=task_timeout)
            elapsed = time.time() - t0
            task.generated_solution = solution
            task.generated_solution_input_tokens = inp_tok
            task.generated_solution_output_tokens = out_tok
            total_input += inp_tok
            total_output += out_tok
            logger.info(
                "  → Done in %.1fs — %d step(s)  tokens: %d in / %d out\n%s",
                elapsed, len(solution), inp_tok, out_tok, str(solution),
            )
        except Exception as e:
            logger.error("  → ERROR on task %s: %s", task.task_ID, e, exc_info=True)

        if output_filename:
            tasks.save_to_file(output_filename, folder=_save_folder)
            logger.debug("  → Intermediate results saved to %s", output_filename)

        if wait_between_tasks:
            time.sleep(wait_between_tasks)

    tasks.metadata["total_input_tokens"] = total_input
    tasks.metadata["total_output_tokens"] = total_output

    if output_filename:
        tasks.save_to_file(output_filename, folder=_save_folder)

    attempted = sum(1 for t in tasks if t.generated_solution is not None)
    degraded = sum(1 for t in tasks if ended_in_error(t.generated_solution))
    logger.info(
        "Generation complete — attempted %d/%d (%d ended in a tool error, "
        "not real solves)  total tokens: %d in / %d out",
        attempted, n_total, degraded, total_input, total_output,
    )
    return tasks, total_input, total_output
