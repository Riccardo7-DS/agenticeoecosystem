"""
Verification run — EVE-Instruct on all runnable tasks (or a curated subset).

Purpose: confirm EVE can call every tool type and chain them correctly.
No LLM evaluators are run; inspect generated solutions in
output/geobenchX/results_EVE-Instruct.json.

Run with:
  python tests/benchmarking.py              # curated subset (default)
  python tests/benchmarking.py --no-subset  # all non-Placeholder tasks
"""

import argparse
import logging

from dotenv import load_dotenv

from agenticeoecosystem.evaluation.dataclasses import TaskSet
from agenticeoecosystem.evaluation.model_comparison import ModelRun, run_comparison
from agenticeoecosystem.utils.logging import setup_logging

load_dotenv()
setup_logging(log_file="benchmarking_eve")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="EVE verification run")
parser.add_argument(
    "--subset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run the curated task subset (default: True). Pass --no-subset for all runnable tasks.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

tasks = TaskSet.read_from_file("eo_benchmark_tasks.json")

if args.subset:
    # Curated subset — one representative task per tool type
    eo_ids = ["TASK_260419_000002_100002", "TASK_260419_000001_100001"]

    # HybridRAG tasks — pure ESA Science Strategy queries (Tool Grounding)
    rag_ids = [
        "TASK_260419_000014_100014", "TASK_260419_000064_100064",
        "TASK_260419_000066_100066", "TASK_260419_000069_100069",
        "TASK_260419_000072_100072", "TASK_260419_000075_100075",
    ]

    # Cross-agent task that chains narratives + GEE + hybridRAG
    cross_ids = ["TASK_260419_000004_100004"]

    selected = tasks.select_by_ids( cross_ids)
    logger.info("Mode: subset — %d tasks selected", len(selected))
else:
    # All tasks whose tools are fully implemented (excludes Placeholder / MOOC)
    selected = tasks.select_by_labels([
        "Tool Grounding",
        "Sequential Reasoning",
        "Cross-agent Synthesis",
        "Scope Awareness",
    ])
    logger.info(
        "Mode: all runnable — %d tasks selected  (breakdown: %s)",
        len(selected),
        selected.get_label_counts(),
    )

# ---------------------------------------------------------------------------
# Generator — EVE only (external API, no vLLM or GPU needed)
# ---------------------------------------------------------------------------

runs = [
    ModelRun(label="EVE-Instruct", model="EVE-Instruct", temperature=0.0, provider="eve"),
]

# ---------------------------------------------------------------------------
# Run — generation only, no LLM evaluation
# ---------------------------------------------------------------------------

df = run_comparison(runs, selected, evaluators=[])

print(df.to_string())
