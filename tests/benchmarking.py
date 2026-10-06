"""
Verification run — EVE-Instruct on all runnable tasks (or a curated subset).

Purpose: confirm EVE can call every tool type and chain them correctly.
No LLM evaluators are run; inspect generated solutions in
output/ESAEOBench/results_EVE-Instruct.json.

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
# Task loading — use the consolidated benchmark
# ---------------------------------------------------------------------------

all_tasks = TaskSet.read_from_file("benchmark_tasks.json")

# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

if args.subset:
    # --- Existing EO-dashboard tasks ---

    # EO Dashboard narrative + indicator tools (Tool Grounding)
    eo_ids = ["TASK_260419_000002_100002", "TASK_260419_000001_100001"]

    # HybridRAG tasks — pure ESA Science Strategy queries (Tool Grounding)
    rag_ids = [
        "TASK_260419_000014_100014", "TASK_260419_000064_100064",
        "TASK_260419_000066_100066", "TASK_260419_000069_100069",
        "TASK_260419_000072_100072", "TASK_260419_000075_100075",
    ]

    # Cross-agent task that chains narratives + GEE + hybridRAG
    cross_ids = ["TASK_260419_000004_100004"]

    # --- Arxiv MCP tasks ---

    # Single-tool: search for EO papers (Tool Grounding)
    arxiv_grounding_ids = [
        "TASK_260427_000001_200001",  # search_papers — SAR flood mapping
        "TASK_260427_000005_200005",  # list_papers
    ]

    # Multi-tool Arxiv chains (Sequential Reasoning)
    arxiv_sequential_ids = [
        "TASK_260427_000002_200002",  # search → download → read
        "TASK_260427_000004_200004",  # search → download → citation_graph
    ]

    # --- GIS MCP tasks ---

    # Single-tool GIS (Tool Grounding)
    gis_grounding_ids = [
        "TASK_260427_000010_200010",  # compute_ndvi
        "TASK_260427_000014_200014",  # transform_coordinates
        "TASK_260427_000016_200016",  # calculate_geodetic_distance
    ]

    # Multi-tool GIS chains (Sequential Reasoning)
    gis_sequential_ids = [
        "TASK_260427_000013_200013",  # compute_ndvi → zonal_statistics
        "TASK_260427_000017_200017",  # buffer → calculate_geodetic_area
    ]

    # --- Cross-server tasks ---

    # ESA Science Strategy RAG + Arxiv
    rag_arxiv_ids = [
        "TASK_260427_000024_200024",  # hybrid_rag_retrieve → search_papers (carbon)
        "TASK_260427_000027_200027",  # hybrid_rag_retrieve → search_papers → get_abstract (water)
    ]

    # GEE download → GIS NDVI pipeline
    gee_gis_ids = [
        "TASK_260427_000031_200031",  # download → compute_ndvi
        "TASK_260427_000033_200033",  # download → compute_ndvi → raster_histogram
        "TASK_260427_000035_200035",  # download → clip → compute_ndvi → zonal_statistics
    ]

    # GEE download → NDVI → map visualisation
    gee_map_ids = [
        "TASK_260427_000037_200037",  # download → compute_ndvi → create_map (raster only)
        "TASK_260427_000039_200039",  # download → compute_ndvi → zonal_statistics → create_map
    ]

    # Scope awareness (control questions — agent must produce no tool calls)
    scope_ids = [
        "TASK_260427_000022_200022",  # restaurants (out-of-scope)
        "TASK_260427_000023_200023",  # crypto trading (out-of-scope)
    ]

    selected_ids = (
        cross_ids
        + arxiv_grounding_ids
        + arxiv_sequential_ids
        + gis_grounding_ids
        + gis_sequential_ids
        + rag_arxiv_ids
        + gee_gis_ids
        + gee_map_ids
        + scope_ids
    )

    selected = all_tasks.select_by_ids(selected_ids)
    logger.info("Mode: subset — %d tasks selected", len(selected))
else:
    # All tasks whose tools are fully implemented (excludes Placeholder / MOOC)
    selected = all_tasks.select_by_labels([
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
