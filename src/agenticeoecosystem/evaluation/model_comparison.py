"""
model_comparison.py — Run the full eval pipeline across multiple models and
produce a side-by-side comparison table (CSV + HTML).

Typical usage
-------------
    from agenticeoecosystem.evaluation.model_comparison import ModelRun, run_comparison
    from agenticeoecosystem.evaluation.dataclasses import TaskSet

    tasks = TaskSet.read_from_file("eo_benchmark_tasks.json")

    runs = [
        ModelRun(label="EVE-Instruct",    model="EVE-Instruct",              provider="eve"),
        ModelRun(label="Qwen2.5-14B",     model="qwen2.5-14b",               provider="vllm"),
        ModelRun(label="Llama3.1-8B",     model="llama3.1-8b",               provider="vllm"),
        ModelRun(label="Mistral-Small",   model="mistral-small-24b",         provider="vllm"),
    ]

    df = run_comparison(runs, tasks, evaluator_model="EVE-Instruct", evaluator_provider="eve")
    print(df)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agenticeoecosystem.definitions import GEOBENCH_RESULTS
from .dataclasses import TaskSet
from .pipeline_runner import generate_solutions, ended_in_error
from .evaluator import evaluate_solutions

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with colored, timestamped output to stdout.
    Call once at the top of your script or notebook before running any comparison.
    """
    from agenticeoecosystem.utils.logging import setup_logging as _setup
    _setup(level)


# ---------------------------------------------------------------------------
# ModelRun descriptor
# ---------------------------------------------------------------------------

@dataclass
class ModelRun:
    """
    Describes one model run to benchmark.

    Args:
        label       : Human-readable name used as a row label in the output table.
        model       : Model identifier passed to generate_solutions / rebuild_agents.
                      For "vllm" provider this can be a registry key (e.g. "qwen2.5-14b")
                      or a raw HuggingFace model ID.
        provider    : LangChain provider — "eve", "openai", "custom", "vllm", "ollama", …
        temperature : Sampling temperature.
        pipeline    : "multi" (google-adk orchestrator, the current multi-agent system —
                      EVE-Instruct only), "single" (one agent, all tools), or
                      "multi_legacy" (older LangGraph supervisor + sub-agents, superseded
                      by "multi").
        agent_models: Per-agent model overrides for pipeline="multi_legacy", e.g.
                      {"gis": ("mistral-small-24b", "vllm"), "arxiv": ("qwen2.5-14b", "vllm")}.
                      Valid keys: eo_stac, gee, eve, eoss_rag, moocs, arxiv, gis, supervisor.
                      Missing keys fall back to (model, provider). Ignored for pipeline="single"
                      or pipeline="multi".
        launch_vllm         : If True, start a local vLLM server for this model before running
                              and shut it down afterwards.  Only meaningful when provider="vllm".
        vllm_port           : Port for the vLLM server (default 8000).  Ignored unless launch_vllm=True.
        vllm_startup_timeout: Seconds to wait for the vLLM /health endpoint before giving up
                              (default 600).  Increase for large models or slow machines.
        extra_meta          : Any extra key/value pairs to include in the result row.
    """
    label: str
    model: str
    provider: str = "eve"
    temperature: float = 0.0
    pipeline: str = "multi"
    agent_models: dict | None = None
    launch_vllm: bool = False
    vllm_port: int = 8000
    vllm_startup_timeout: int = 600
    extra_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def _launch_vllm_for_run(run: ModelRun, label: str, role: str) -> Optional[subprocess.Popen]:
    """Launch a vLLM server for a ModelRun if launch_vllm=True. Returns the process or None."""
    if not run.launch_vllm:
        return None
    from agenticeoecosystem.utils.registry import launch_vllm
    logger.info("[%s] Launching %s vLLM (%s) on port %d (timeout=%ds) …",
                label, role, run.model, run.vllm_port, run.vllm_startup_timeout)
    proc = launch_vllm(run.model, port=run.vllm_port, startup_timeout=run.vllm_startup_timeout)
    os.environ["VLLM_PORT"] = str(run.vllm_port)
    return proc


def _terminate(proc: Optional[subprocess.Popen], label: str, role: str) -> None:
    if proc is None:
        return
    import signal
    logger.info("[%s] Terminating %s vLLM process group", label, role)
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # Workers didn't exit on SIGTERM — force-kill the whole group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Phase 1: generation
# ---------------------------------------------------------------------------

def _generate_one(
    run: ModelRun,
    tasks: TaskSet,
    output_dir: Path,
    skip_solved: bool,
) -> tuple[TaskSet, dict]:
    """
    Run the agent pipeline for one ModelRun and return the populated TaskSet
    plus a metadata dict that _evaluate_one needs.

    Raises on any error — caller catches and records it.
    """
    label = run.label
    task_copy = copy.deepcopy(tasks)
    t_run_start = time.time()
    proc_gen: Optional[subprocess.Popen] = None

    try:
        proc_gen = _launch_vllm_for_run(run, label, "generator")
        output_filename = f"results_{label.replace(' ', '_')}.json"
        logger.info("[%s] ── Generating solutions …", label)
        t0 = time.time()
        task_copy, in_tok, out_tok = generate_solutions(
            task_copy,
            model=run.model,
            temperature=run.temperature,
            model_provider=run.provider,
            output_filename=output_filename,
            output_folder=str(output_dir),
            skip_solved=skip_solved,
            pipeline=run.pipeline,
            agent_models=run.agent_models,
        )
        logger.info("[%s] Generation done in %.1fs", label, time.time() - t0)
    finally:
        _terminate(proc_gen, label, "generator")
        if run.launch_vllm:
            time.sleep(3)

    return task_copy, {
        "output_filename": output_filename,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "t_run_start": t_run_start,
    }


# ---------------------------------------------------------------------------
# Phase 2: evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    run: ModelRun,
    task_copy: TaskSet,
    gen_meta: dict,
    evaluators: list,
    output_dir: Path,
    skip_evaluated: bool,
) -> dict:
    """
    Score a generated TaskSet with every evaluator and return a complete result row.

    Raises on any error — caller catches and records it.
    """
    label = run.label
    output_filename = gen_meta["output_filename"]
    in_tok = gen_meta["in_tok"]
    out_tok = gen_meta["out_tok"]
    t_run_start = gen_meta["t_run_start"]

    for eval_run in evaluators:
        proc_eval: Optional[subprocess.Popen] = None
        try:
            logger.info("[%s] ── Evaluating with %s (%s) …", label, eval_run.label, eval_run.provider)
            proc_eval = _launch_vllm_for_run(eval_run, label, f"evaluator[{eval_run.label}]")

            if eval_run.launch_vllm:
                from agenticeoecosystem.utils.registry import get_vllm_model_name
                model_id = get_vllm_model_name(port=eval_run.vllm_port)
                logger.info("[%s] Evaluator vLLM serving model as: %s", label, model_id)
            else:
                model_id = eval_run.model

            os.environ["EVALUATOR_PROVIDER"] = eval_run.provider
            t0 = time.time()
            task_copy = evaluate_solutions(
                task_copy,
                model=model_id,
                output_filename=output_filename,
                skip_evaluated=skip_evaluated,
                evaluator_label=eval_run.label,
            )
            logger.info("[%s] Evaluation with %s done in %.1fs", label, eval_run.label, time.time() - t0)

        finally:
            _terminate(proc_eval, label, f"evaluator[{eval_run.label}]")
            if eval_run.launch_vllm:
                time.sleep(3)

    task_copy.save_to_file(output_filename, folder=str(output_dir))
    logger.info("[%s] Results saved → %s/%s", label, output_dir, output_filename)

    # Build result row.
    # "tasks_attempted" counts any run that produced a generated_solution at
    # all; "tasks_degraded" is the subset of those where the trace's last
    # step is a tool call that errored out (e.g. an MCP timeout or 404 left
    # as the final step) rather than a real answer — these get silently
    # counted as solved unless called out separately.
    attempted = sum(1 for t in task_copy if t.generated_solution is not None)
    degraded = sum(1 for t in task_copy if ended_in_error(t.generated_solution))
    n_steps_list = [len(t.generated_solution) for t in task_copy if t.generated_solution]
    avg_steps = sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0.0

    row: dict = {
        "model": label,
        "provider": run.provider,
        "pipeline": run.pipeline,
        "tasks_attempted": attempted,
        "tasks_degraded": degraded,
        "avg_steps": round(avg_steps, 1),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "elapsed_s": round(time.time() - t_run_start, 1),
    }

    for eval_run in evaluators:
        ev = task_copy.score_summary_for_evaluator(eval_run.label)
        elabel = eval_run.label
        row[f"evaluated [{elabel}]"] = ev.get("n", 0)
        row[f"match (2) [{elabel}]"] = ev.get("match", 0)
        row[f"partial (1) [{elabel}]"] = ev.get("partial", 0)
        row[f"no_match (0) [{elabel}]"] = ev.get("no_match", 0)
        row[f"accuracy_full [{elabel}]"] = round(ev.get("accuracy_full", 0.0), 3)
        row[f"accuracy_partial+full [{elabel}]"] = round(ev.get("accuracy_partial_or_full", 0.0), 3)
        logger.info(
            "[%s] ── Summary [%s] — evaluated: %d  match: %d  partial: %d  no_match: %d  accuracy: %.1f%%",
            label, elabel, ev.get("n", 0),
            ev.get("match", 0), ev.get("partial", 0), ev.get("no_match", 0),
            ev.get("accuracy_full", 0.0) * 100,
        )

    logger.info("[%s] ── Total elapsed: %.1fs", label, time.time() - t_run_start)
    row.update(run.extra_meta)
    return row


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    runs: list[ModelRun],
    tasks: TaskSet,
    evaluators: list[ModelRun],
    output_dir: Optional[Path] = None,
    skip_solved: bool = False,
    skip_evaluated: bool = True,
    table_filename: str = "model_comparison",
    smoke_test_labels: Optional[list[str]] = None,
    smoke_test_n: int = 2,
) -> Any:
    """
    Run the full eval pipeline for every generator ModelRun, scoring each with
    every evaluator ModelRun.  Returns a DataFrame with per-evaluator columns.

    Generation and evaluation are run in two separate passes: all generators
    complete before any evaluation begins, so results are not mixed across phases.

    Args:
        runs               : Generator models to benchmark.
        tasks              : TaskSet to evaluate on (deep-copied per run).
        evaluators         : List of LLM-as-judge models.  Each ModelRun can set
                             launch_vllm=True / vllm_startup_timeout as needed.
                             Do not use EVE as an evaluator.
        output_dir         : Folder for all output files (default GEOBENCH_RESULTS).
        skip_solved        : If True, skip tasks that already have a generated solution.
        skip_evaluated     : If True, skip tasks already scored by that evaluator.
        table_filename     : Base filename (no extension) for CSV and HTML outputs.
        smoke_test_labels  : Filter to tasks matching these labels before running.
        smoke_test_n       : Max tasks to keep after label filtering (default 2).
    """
    import pandas as pd

    if output_dir is None:
        output_dir = Path(GEOBENCH_RESULTS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if smoke_test_labels is not None:
        tasks = tasks.select_by_labels(smoke_test_labels)
        tasks = TaskSet(metadata=tasks.metadata, tasks=tasks.tasks[:smoke_test_n])
        logger.info(
            "Smoke-test mode: %d task(s) kept (labels=%s, n=%d)",
            len(tasks), smoke_test_labels, smoke_test_n,
        )
        if len(tasks) == 0:
            raise ValueError(
                f"No tasks found for labels {smoke_test_labels}. "
                "Check your task file or label spelling."
            )
        table_filename = f"{table_filename}_smoke"

    gen_labels  = [r.label for r in runs]
    eval_labels = [e.label for e in evaluators]
    logger.info("╔══ run_comparison ══════════════════════════════════════════")
    logger.info("║  generators : %s", ", ".join(gen_labels))
    logger.info("║  evaluators : %s", ", ".join(eval_labels))
    logger.info("║  tasks      : %d  |  output → %s", len(tasks), output_dir)
    logger.info("╚════════════════════════════════════════════════════════════")

    errors: list[dict] = []
    t_total_start = time.time()

    # ── Phase 1: generate solutions for every model ──────────────────────────
    logger.info("━━ Phase 1/2: Generating solutions (%d model(s)) ━━━━━━━━━━━━", len(runs))
    gen_results: dict[str, tuple[TaskSet, dict]] = {}

    for idx, run in enumerate(runs, 1):
        logger.info("┌── [Generate] %d/%d: %s", idx, len(runs), run.label)
        try:
            task_copy, gen_meta = _generate_one(run, tasks, output_dir, skip_solved)
            gen_results[run.label] = (task_copy, gen_meta)
            logger.info("└── [Generate] %d/%d complete", idx, len(runs))
        except Exception as exc:
            logger.error(
                "└── [Generate] %d/%d FAILED (%s): %s", idx, len(runs), run.label, exc,
                exc_info=True,
            )
            errors.append({"model": run.label, "phase": "generate", "error": str(exc)})

    # ── Phase 2: evaluate all generated solutions ─────────────────────────────
    succeeded = [r for r in runs if r.label in gen_results]
    logger.info("━━ Phase 2/2: Evaluating solutions (%d model(s)) ━━━━━━━━━━━━", len(succeeded))
    rows: list[dict] = []

    for idx, run in enumerate(succeeded, 1):
        task_copy, gen_meta = gen_results[run.label]
        logger.info("┌── [Evaluate] %d/%d: %s", idx, len(succeeded), run.label)
        try:
            row = _evaluate_one(run, task_copy, gen_meta, evaluators, output_dir, skip_evaluated)
            rows.append(row)
            logger.info("└── [Evaluate] %d/%d complete  elapsed=%.1fs", idx, len(succeeded), row["elapsed_s"])
        except Exception as exc:
            logger.error(
                "└── [Evaluate] %d/%d FAILED (%s): %s", idx, len(succeeded), run.label, exc,
                exc_info=True,
            )
            errors.append({"model": run.label, "phase": "evaluate", "error": str(exc)})

    # ── Save comparison table ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{table_filename}.csv"
    html_path = output_dir / f"{table_filename}.html"
    df.to_csv(csv_path, index=False)
    df.to_html(html_path, index=False)
    logger.info("Comparison table saved → %s (CSV + HTML)", output_dir / table_filename)
    logger.info("Total elapsed: %.1fs", time.time() - t_total_start)

    if errors:
        errors_path = output_dir / f"{table_filename}_errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning("%d run(s) failed — see %s", len(errors), errors_path)

    return df
