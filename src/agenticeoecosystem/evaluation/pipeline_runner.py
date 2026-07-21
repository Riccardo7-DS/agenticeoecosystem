"""
pipeline_runner.py — Runs your agentic pipeline on a TaskSet and records results.
"""

import asyncio
import logging
import time
from typing import Optional

from tqdm import tqdm

from agenticeoecosystem.definitions import GEOBENCH_RESULTS
from .dataclasses import Solution, Step, TaskSet

logger = logging.getLogger(__name__)


def run_pipeline(
    task_text: str,
    pipeline: str = "multi",
) -> tuple[Solution, int, int]:
    """
    Run the agent pipeline on a single task and return the MCP-level tool steps.

    Args:
        task_text: The natural-language task prompt.
        pipeline : "multi" (LangGraph supervisor + sub-agents), "single" (one
                   LangGraph agent, all tools), or "adk" (google-adk
                   orchestrator + native sub-agents, see agents/multi_agent.py).

    Returns:
        solution     : Solution with one Step per MCP tool call made.
        input_tokens : Total input tokens across all agents.
        output_tokens: Total output tokens across all agents.
    """
    import nest_asyncio
    nest_asyncio.apply()

    if pipeline == "adk":
        import json as _json
        import agenticeoecosystem.agents.multi_agent as _adk

        loop = asyncio.get_event_loop()
        raw = loop.run_until_complete(_adk.get_ans_tool_path(task_text))
        data = _json.loads(raw)
        steps = [
            Step(function_name=t["function_name"], arguments=t.get("arguments", {}))
            for t in data.get("tools_called", [])
        ]
        return (
            Solution(steps=steps),
            data.get("input_tokens", 0),
            data.get("output_tokens", 0),
        )

    if pipeline == "single":
        import agenticeoecosystem.client.client_single_agent as _module
        _agent = _module.agent
    else:
        import agenticeoecosystem.client.client as _module
        _agent = _module.supervisor_agent

    _module.reset_step_log()

    async def _run() -> tuple[int, int]:
        async for chunk in _agent.astream(
            {"messages": [{"role": "user", "content": task_text}]},
            config={"recursion_limit": 50},
        ):
            for update in chunk.values():
                for msg in update.get("messages", []):
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
        pipeline         : "multi" (LangGraph supervisor + sub-agents), "single"
                           (one LangGraph agent, all tools), or "adk" (google-adk
                           orchestrator, always runs EVE-Instruct — model/provider
                           overrides are not yet supported for this pipeline).
        agent_models     : Per-agent model overrides for pipeline="multi", e.g.
                           {"gis": ("mistral-small-24b", "vllm")}. Ignored for
                           pipeline="single" or pipeline="adk".

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
    elif pipeline == "adk":
        if agent_models:
            logger.warning("agent_models is ignored when pipeline='adk'")
        if model != "EVE-Instruct" or model_provider != "eve":
            logger.warning(
                "pipeline='adk' does not support model/provider overrides yet — "
                "it always runs EVE-Instruct via the EVE endpoint regardless of "
                "model=%s provider=%s", model, model_provider,
            )
        import agenticeoecosystem.agents.multi_agent  # noqa: F401 (import triggers setup)
    else:
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
            solution, inp_tok, out_tok = run_pipeline(task.task_text, pipeline=pipeline)
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

    logger.info(
        "Generation complete — solved %d/%d  total tokens: %d in / %d out",
        sum(1 for t in tasks if t.generated_solution is not None),
        n_total, total_input, total_output,
    )
    return tasks, total_input, total_output
