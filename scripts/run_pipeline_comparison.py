#!/usr/bin/env python
"""
run_pipeline_comparison.py — Compare single-agent vs multi-agent on the same task set.

Runs generate_solutions twice with the same model (once per pipeline architecture),
evaluates both with the specified evaluator, and saves a side-by-side comparison table.

Example — full benchmark:
    python scripts/run_pipeline_comparison.py \
        --model EVE-Instruct \
        --provider eve \
        --tasks-file eo_benchmark_tasks.json arxiv_gis_benchmark_tasks.json \
        --evaluator-model Nemotron-2-12B \
        --evaluator-provider eve \
        --output-dir output/pipeline_comparison

Example — smoke test (2 tasks, Tool Grounding only):
    python scripts/run_pipeline_comparison.py \
        --model EVE-Instruct \
        --provider eve \
        --tasks-file eo_benchmark_tasks.json \
        --evaluator-model Nemotron-2-12B \
        --evaluator-provider eve \
        --output-dir output/pipeline_comparison \
        --smoke-test

Example — multi-agent with per-agent model overrides:
    python scripts/run_pipeline_comparison.py \
        --model EVE-Instruct \
        --provider eve \
        --agent-models '{"gis": ["mistral-small-24b", "vllm"], "arxiv": ["qwen2.5-14b", "vllm"]}' \
        --tasks-file eo_benchmark_tasks.json \
        --evaluator-model Nemotron-2-12B \
        --evaluator-provider eve \
        --output-dir output/pipeline_comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare single-agent vs multi-agent pipeline on the same task set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",              required=True,  help="Model key or HuggingFace ID")
    parser.add_argument("--provider",           default="eve",  help="LangChain provider: eve, vllm, openai, ollama, …")
    parser.add_argument("--temperature",        type=float, default=0.0)
    parser.add_argument("--agent-models",       default=None,
                        help='JSON per-agent model overrides applied to the multi-agent run only. '
                             'e.g. \'{"gis": ["mistral-small-24b", "vllm"], "arxiv": ["qwen2.5-14b", "vllm"]}\'')
    parser.add_argument("--tasks-file",         required=True, nargs="+",
                        help="Tasks JSON filename(s) inside --tasks-folder; multiple files are merged")
    parser.add_argument("--tasks-folder",       default=None,
                        help="Folder containing task files (default: data/tasks)")
    parser.add_argument("--task-labels",        default=None,
                        help="Comma-separated task labels to filter, e.g. 'Tool Grounding,Sequential Reasoning'")
    parser.add_argument("--evaluator-model",    required=True,  help="Evaluator model key or HuggingFace ID")
    parser.add_argument("--evaluator-provider", default="eve",  help="Evaluator provider: eve, vllm, openai, …")
    parser.add_argument("--output-dir",         required=True,  help="Directory for results JSON and comparison table")
    parser.add_argument("--table-filename",     default="pipeline_comparison",
                        help="Base filename (no extension) for CSV and HTML outputs")
    parser.add_argument("--skip-solved",        action="store_true",
                        help="Skip tasks that already have a generated solution in an existing results file")
    parser.add_argument("--smoke-test",         action="store_true",
                        help="Quick run: keep only 2 Tool Grounding tasks per pipeline")
    args = parser.parse_args()

    setup_logging(log_file="pipeline_comparison")
    logger = logging.getLogger(__name__)

    from agenticeoecosystem.definitions import TASKS_FOLDER
    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    from agenticeoecosystem.evaluation.model_comparison import ModelRun, run_comparison

    # Load and merge task files
    folder = args.tasks_folder or str(TASKS_FOLDER)
    tasks: TaskSet | None = None
    for tf in args.tasks_file:
        ts = TaskSet.read_from_file(tf, folder=folder)
        if tasks is None:
            tasks = ts
        else:
            tasks.extend(ts.tasks)

    if args.task_labels:
        labels = [x.strip() for x in args.task_labels.split(",")]
        tasks = tasks.select_by_labels(labels)

    logger.info("Loaded %d task(s) from %d file(s)", len(tasks), len(args.tasks_file))

    # Parse per-agent model overrides (multi pipeline only)
    agent_models = None
    if args.agent_models:
        raw = json.loads(args.agent_models)
        agent_models = {k: tuple(v) for k, v in raw.items()}
        logger.info("Per-agent overrides (multi only): %s", agent_models)

    # Build the two pipeline runs with the same base model
    multi_label  = f"{args.model} [multi]"
    single_label = f"{args.model} [single]"

    runs = [
        ModelRun(
            label=multi_label,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            pipeline="multi",
            agent_models=agent_models,
        ),
        ModelRun(
            label=single_label,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            pipeline="single",
        ),
    ]

    evaluators = [
        ModelRun(
            label=args.evaluator_model,
            model=args.evaluator_model,
            provider=args.evaluator_provider,
        )
    ]

    smoke_labels = ["Tool Grounding"] if args.smoke_test else None

    df = run_comparison(
        runs=runs,
        tasks=tasks,
        evaluators=evaluators,
        output_dir=Path(args.output_dir),
        skip_solved=args.skip_solved,
        table_filename=args.table_filename,
        smoke_test_labels=smoke_labels,
        smoke_test_n=2,
    )

    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
