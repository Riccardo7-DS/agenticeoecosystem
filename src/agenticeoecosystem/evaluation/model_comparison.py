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
from .pipeline_runner import generate_solutions
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
    Describes one model to benchmark.

    Args:
        label       : Human-readable name used as a row label in the output table.
        model       : Model identifier passed to generate_solutions / rebuild_agents.
                      For "vllm" provider this can be a registry key (e.g. "qwen2.5-14b")
                      or a raw HuggingFace model ID.
        provider    : LangChain provider — "eve", "openai", "custom", "vllm", "ollama", …
        temperature : Sampling temperature.
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
    launch_vllm: bool = False
    vllm_port: int = 8000
    vllm_startup_timeout: int = 600
    extra_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Single-model runner
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
    logger.info("[%s] Terminating %s vLLM process", label, role)
    proc.terminate()
    proc.wait(timeout=30)


def _run_one(
    run: ModelRun,
    tasks: TaskSet,
    evaluators: list,   # list[ModelRun]
    output_dir: Path,
    skip_solved: bool,
    skip_evaluated: bool,
) -> dict:
    """Run generate then evaluate with every evaluator for one ModelRun."""
    label = run.label
    sep = "=" * 60
    logger.info("%s", sep)
    logger.info("MODEL RUN: %s  (provider=%s  model=%s  temp=%.1f)", label, run.provider, run.model, run.temperature)
    logger.info("%s", sep)

    task_copy = copy.deepcopy(tasks)
    t_run_start = time.time()
    proc_gen: Optional[subprocess.Popen] = None

    try:
        # --- Phase 1: Generate ---
        proc_gen = _launch_vllm_for_run(run, label, "generator")
        output_filename = f"results_{label.replace(' ', '_')}.json"
        logger.info("[%s] ── Phase 1/%d: Generating solutions ──────────────────",
                    label, 1 + len(evaluators))
        t0 = time.time()
        task_copy, in_tok, out_tok = generate_solutions(
            task_copy,
            model=run.model,
            temperature=run.temperature,
            model_provider=run.provider,
            output_filename=output_filename,
            skip_solved=skip_solved,
        )
        logger.info("[%s] Generation done in %.1fs", label, time.time() - t0)

        # Terminate generator before evaluators to free GPU/port
        _terminate(proc_gen, label, "generator")
        proc_gen = None
        if run.launch_vllm:
            time.sleep(3)

        # --- Phase 2+: Evaluate with each evaluator ---
        for eval_idx, eval_run in enumerate(evaluators, 2):
            proc_eval: Optional[subprocess.Popen] = None
            try:
                logger.info("[%s] ── Phase %d/%d: Evaluating with %s (%s) ──────────────",
                            label, eval_idx, 1 + len(evaluators), eval_run.label, eval_run.provider)
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

        # --- Save ---
        task_copy.save_to_file(output_filename, folder=str(output_dir))
        logger.info("[%s] Results saved → %s/%s", label, output_dir, output_filename)

    finally:
        _terminate(proc_gen, label, "generator")

    # --- Extract metrics ---
    solved = sum(1 for t in task_copy if t.generated_solution is not None)
    n_steps_list = [len(t.generated_solution) for t in task_copy if t.generated_solution]
    avg_steps = sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0.0

    row: dict = {
        "model": label,
        "provider": run.provider,
        "tasks_attempted": solved,
        "avg_steps": round(avg_steps, 1),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "elapsed_s": round(time.time() - t_run_start, 1),
    }

    # Per-evaluator accuracy columns
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

    rows = []
    errors = []
    t_total_start = time.time()

    for idx, run in enumerate(runs, 1):
        logger.info("┌── Run %d/%d: %s", idx, len(runs), run.label)
        try:
            row = _run_one(
                run=run,
                tasks=tasks,
                evaluators=evaluators,
                output_dir=output_dir,
                skip_solved=skip_solved,
                skip_evaluated=skip_evaluated,
            )
            rows.append(row)
            logger.info("└── Run %d/%d complete  elapsed=%.1fs", idx, len(runs), row["elapsed_s"])
        except Exception as exc:
            logger.error("└── Run %d/%d FAILED (%s): %s", idx, len(runs), run.label, exc, exc_info=True)
            errors.append({"model": run.label, "error": str(exc)})

    import pandas as pd
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
