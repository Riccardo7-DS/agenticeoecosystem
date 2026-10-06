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
from agenticeoecosystem.utils.pricing import estimate_cost_usd
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
                tool_result=t.get("tool_result"),
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
    final_answer: str | None = None

    async def _run() -> tuple[int, int]:
        nonlocal final_answer
        final_answer = None
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
                    for msg in messages:
                        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
                            content = msg.content
                            if isinstance(content, str):
                                final_answer = content
                            elif isinstance(content, list):
                                final_answer = "\n".join(
                                    block if isinstance(block, str) else block.get("text", "")
                                    for block in content
                                    if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
                                )
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

    def _is_dead_session_error(exc: BaseException) -> bool:
        # Matched by class name rather than importing anyio directly, since
        # this module is pipeline-agnostic and anyio is only a dependency of
        # the "single"/"multi_legacy" MCP client stack. Walks __cause__ and
        # __context__ too — the error can arrive wrapped depending on which
        # layer (langgraph's tool_node, langchain's BaseTool.arun, etc.)
        # last touched it.
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if type(exc).__name__ == "ClosedResourceError":
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    def _run_once():
        coro = _run()
        if task_timeout:
            coro = asyncio.wait_for(coro, timeout=task_timeout)
        return loop.run_until_complete(coro)

    try:
        in_tok, out_tok = _run_once()
    except Exception as exc:
        if _is_dead_session_error(exc) and hasattr(_module, "reconnect_mcp_tools"):
            # A persistent MCP session (e.g. gis-mcp) died mid-run, poisoning
            # every subsequent call to that server for the rest of the
            # process — see reconnect_mcp_tools's docstring. Rebuild once and
            # retry this task; if it still fails, let the exception surface
            # normally (this task recorded as unsolved) rather than loop.
            logger.warning(
                "run_pipeline: dead MCP session detected (%s) — reconnecting and retrying this task once",
                exc,
            )
            _module.reconnect_mcp_tools()
            _agent = _module.agent
            _module.reset_step_log()
            in_tok, out_tok = _run_once()
        else:
            raise

    steps = [
        Step(function_name=s["function_name"], arguments=s["arguments"], tool_result=s.get("tool_result"))
        for s in _module._step_log
    ]
    return Solution(steps=steps, final_answer=final_answer), in_tok, out_tok


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
    archive_evidence: bool = False,
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
                           server, "openai"/"openrouter"/"mistral" route to the
                           real, billed API, any other value falls back to
                           "eve"), "single" (one LangGraph agent, all tools), or
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
    run_start = time.time()
    healthcheck_results: dict = {}
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
        if model_provider not in ("eve", "vllm", "openai", "openrouter", "mistral"):
            logger.warning(
                "pipeline='multi' only supports provider='eve' (internal EVE "
                "gateway), 'vllm' (local server), 'openai' (real OpenAI API, "
                "real billing), 'openrouter' (real OpenRouter API, real "
                "billing), or 'mistral' (real Mistral API, real billing) — "
                "ignoring model_provider=%s, falling back to 'eve'",
                model_provider,
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

        from agenticeoecosystem.utils.healthcheck import run_preflight_healthcheck
        healthcheck_results = asyncio.get_event_loop().run_until_complete(run_preflight_healthcheck())
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
    total_cost = estimate_cost_usd(model, 0, 0)  # None if `model` has no known rate card

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
            task.candidate_artifacts = []
            task.generated_solution_input_tokens = inp_tok
            task.generated_solution_output_tokens = out_tok
            if archive_evidence:
                from pathlib import Path
                from .candidate_evidence import CandidateEvidence
                from agenticeoecosystem.definitions import ROOT_DIR
                CandidateEvidence.archive(task, (ROOT_DIR / "output", ROOT_DIR / "tool_output"),
                                          Path(_save_folder) / "evidence" / Path(output_filename or model).stem)
            total_input += inp_tok
            total_output += out_tok
            task_cost = estimate_cost_usd(model, inp_tok, out_tok)
            if task_cost is not None:
                total_cost += task_cost
                logger.info(
                    "  → Done in %.1fs — %d step(s)  tokens: %d in / %d out  "
                    "cost: $%.4f (running total: $%.2f)\n%s",
                    elapsed, len(solution), inp_tok, out_tok, task_cost, total_cost, str(solution),
                )
            else:
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
    if total_cost is not None:
        tasks.metadata["total_cost_usd"] = round(total_cost, 4)

    if output_filename:
        tasks.save_to_file(output_filename, folder=_save_folder)

    attempted = sum(1 for t in tasks if t.generated_solution is not None)
    degraded = sum(1 for t in tasks if ended_in_error(t.generated_solution))
    cost_str = f"  total cost: ${total_cost:.2f}" if total_cost is not None else ""
    logger.info(
        "Generation complete — attempted %d/%d (%d ended in a tool error, "
        "not real solves)  total tokens: %d in / %d out%s",
        attempted, n_total, degraded, total_input, total_output, cost_str,
    )

    _write_run_summary(
        tasks, model, pipeline, healthcheck_results, run_start,
        n_total, attempted, n_skip, degraded, total_input, total_output,
        output_filename, _save_folder,
    )

    return tasks, total_input, total_output


def _write_run_summary(
    tasks: TaskSet, model: str, pipeline: str, healthcheck_results: dict,
    run_start: float, n_total: int, attempted: int, n_skip: int, degraded: int,
    total_input: int, total_output: int, output_filename: Optional[str], save_folder: str,
) -> None:
    """Build the clean run-overview block: healthcheck, duration, task counts,
    tokens overall and by label. Logged as one bordered block and also written
    to a standalone summary_<label>.md next to (but separate from) the
    results_<label>.json used for evaluation."""
    from agenticeoecosystem.utils.healthcheck import log_block

    elapsed = int(time.time() - run_start)
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    duration_str = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"

    lines = []
    if healthcheck_results:
        n_ok = sum(healthcheck_results.values())
        lines.append(f"Healthcheck: {n_ok}/{len(healthcheck_results)} reachable")
        for name, ok in healthcheck_results.items():
            if not ok:
                lines.append(f"  FAIL  {name}")
    lines.append(f"Duration: {duration_str}")
    lines.append(
        f"Tasks: {n_total} total — {attempted} attempted, {n_skip} skipped "
        f"(already solved), {degraded} ended in a tool error"
    )
    lines.append(f"Tokens: {total_input} in / {total_output} out (total)")

    label_totals = tasks.get_token_totals_by_label()
    if label_totals:
        lines.append("Tokens by task label:")
        for label, stats in sorted(label_totals.items(), key=lambda kv: str(kv[0])):
            label_str = label.value if hasattr(label, "value") else label
            lines.append(
                f"  {label_str}: {stats['n']} task(s) — {stats['input']} in / {stats['output']} out"
            )

    log_block(f"RUN SUMMARY — {model} ({pipeline})", lines)

    # Separate, properly-formatted Markdown for the .md file — the plain
    # indented `lines` above read fine in a bordered log block, but a single
    # newline doesn't break a line in Markdown, so writing them as-is
    # collapses into one run-on paragraph in any Markdown preview.
    md: list[str] = [f"# Run summary — {model} ({pipeline})", ""]

    if healthcheck_results:
        n_ok = sum(healthcheck_results.values())
        md.append(f"**Healthcheck:** {n_ok}/{len(healthcheck_results)} reachable")
        failed = [name for name, ok in healthcheck_results.items() if not ok]
        if failed:
            md.append("")
            md.extend(f"- FAIL {name}" for name in failed)
        md.append("")

    md.append(f"**Duration:** {duration_str}")
    md.append("")
    md.append(
        f"**Tasks:** {n_total} total — {attempted} attempted, {n_skip} skipped "
        f"(already solved), {degraded} ended in a tool error"
    )
    md.append("")
    md.append(f"**Tokens:** {total_input} in / {total_output} out (total)")

    if label_totals:
        md.append("")
        md.append("**Tokens by task label:**")
        md.append("")
        md.append("| Label | Tasks | Tokens in | Tokens out |")
        md.append("|---|---|---|---|")
        for label, stats in sorted(label_totals.items(), key=lambda kv: str(kv[0])):
            label_str = label.value if hasattr(label, "value") else label
            md.append(f"| {label_str} | {stats['n']} | {stats['input']} | {stats['output']} |")

    if output_filename:
        summary_filename = os.path.splitext(output_filename.replace("results_", "summary_", 1))[0] + ".md"
    else:
        summary_filename = f"summary_{model}_{pipeline}.md".replace(" ", "_")
    os.makedirs(save_folder, exist_ok=True)
    summary_path = os.path.join(save_folder, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    logger.info("Run summary written → %s", summary_path)
