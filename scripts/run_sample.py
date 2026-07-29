#!/usr/bin/env python
"""
run_sample.py — Quick look at a handful of benchmark tasks before scheduling
a full run_generate.py / benchmark.sh job.

Picks a small number of tasks from a tasks JSON file, runs each one through
a pipeline right now in this process, and prints the tool-call trace + final
answer for each — no LLM-as-judge scoring, no results file written. This is
for "does this look sane" sanity checks, not for recording results.

Examples:
    # 5 random Tool-Grounding tasks through the single-agent ADK baseline
    python scripts/run_sample.py --tasks-file benchmark_tasks.json \\
        --pipeline single_adk --task-labels "Tool Grounding" --n 5 --random

    # A specific handful of tasks by ID, through the multi-agent orchestrator
    python scripts/run_sample.py --tasks-file benchmark_tasks.json \\
        --pipeline multi \\
        --task-ids TASK_260419_000001_100001,TASK_260419_000002_100002

    # First 3 tasks overall, single_adk, also print the reference solution
    python scripts/run_sample.py --tasks-file benchmark_tasks.json \\
        --pipeline single_adk --n 3 --show-reference
"""

import argparse
import logging
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def _format_steps(steps) -> str:
    lines = []
    for step in steps:
        args_str = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
        line = f"  - {step.function_name}({args_str})"
        if step.comment:
            line += f"   # {step.comment}"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no tool calls)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small sample of benchmark tasks for a quick look before scheduling a full job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tasks-file", required=True, nargs="+",
                         help="Tasks JSON filename(s) under data/tasks (or --tasks-folder); multiple files are merged")
    parser.add_argument("--tasks-folder", default=None, help="Folder containing tasks file (default: data/tasks)")
    parser.add_argument("--pipeline", default="single_adk",
                         choices=["multi", "single_adk", "single", "multi_legacy"],
                         help="'multi' (ADK orchestrator + sub-agents), 'single_adk' (ADK, one "
                              "agent, all tools — agents/single_agent.py), 'single' (LangChain, "
                              "one agent, sweepable across any model), or 'multi_legacy'. "
                              "Default: single_adk")
    parser.add_argument("--model", default="EVE-Instruct",
                         help="Only used for --pipeline single/multi_legacy (multi/single_adk always use EVE-Instruct)")
    parser.add_argument("--provider", default="eve", help="Only used for --pipeline single/multi_legacy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs to run (takes priority over --task-labels)")
    parser.add_argument("--task-labels", default=None,
                         help="Comma-separated task labels to filter to first, e.g. 'Tool Grounding,Scope Awareness'")
    parser.add_argument("--n", type=int, default=5, help="How many tasks to run after any filter (default: 5)")
    parser.add_argument("--random", action="store_true", help="Pick --n tasks at random instead of the first --n")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --random (default: unseeded)")
    parser.add_argument("--show-reference", action="store_true", help="Also print each task's reference solution(s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from agenticeoecosystem.definitions import TASKS_FOLDER
    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    from agenticeoecosystem.evaluation.pipeline_runner import run_pipeline

    folder = args.tasks_folder or str(TASKS_FOLDER)
    tasks = None
    for tf in args.tasks_file:
        ts = TaskSet.read_from_file(tf, folder=folder)
        if tasks is None:
            tasks = ts
        else:
            tasks.extend(ts.tasks)
    logger.info("Loaded %d task(s) from %d file(s) in %s", len(tasks), len(args.tasks_file), folder)

    if args.task_ids:
        ids = [x.strip() for x in args.task_ids.split(",")]
        tasks = tasks.select_by_ids(ids)
    elif args.task_labels:
        labels = [x.strip() for x in args.task_labels.split(",")]
        tasks = tasks.select_by_labels(labels)

    pool = list(tasks.tasks)
    if not pool:
        logger.error("No tasks matched the given filters — nothing to run.")
        return 1

    if args.random:
        rng = random.Random(args.seed)
        sample = rng.sample(pool, min(args.n, len(pool)))
    else:
        sample = pool[: args.n]

    print(f"\nSelected {len(sample)}/{len(pool)} matching task(s) — pipeline={args.pipeline}\n")

    # For multi/single_adk, model is always EVE-Instruct via the shared
    # MULTI_AGENT_MODEL env var — set it before the first import in this
    # process (see pipeline_runner.generate_solutions for the same pattern).
    import os
    if args.pipeline in ("multi", "single_adk"):
        os.environ.setdefault("MULTI_AGENT_MODEL", args.model)

    for i, task in enumerate(sample, 1):
        print("=" * 100)
        print(f"[{i}/{len(sample)}] {task.task_ID}  {[l.value for l in (task.task_labels or [])]}")
        print(f"Q: {task.task_text}")
        print("-" * 100)

        try:
            solution, in_tok, out_tok = run_pipeline(task.task_text, pipeline=args.pipeline)
        except Exception as exc:
            logger.error("Task %s FAILED: %s", task.task_ID, exc, exc_info=True)
            print(f"FAILED: {exc}")
            continue

        print(f"Tool calls ({len(solution.steps)}):")
        print(_format_steps(solution.steps))
        print(f"\nFinal answer:\n{solution.final_answer}")
        print(f"\nTokens: {in_tok} in / {out_tok} out")

        if args.show_reference and task.reference_solutions:
            print("\nReference solution(s):")
            for j, ref in enumerate(task.reference_solutions, 1):
                print(f"  [{j}]")
                print(_format_steps(ref.steps))

    print("=" * 100)
    print(f"Done: ran {len(sample)} task(s) with pipeline={args.pipeline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
