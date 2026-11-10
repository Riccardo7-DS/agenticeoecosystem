"""Helper settings.

MODE is set once at runtime by helper.configure(); everything else is a default.
    off       no recording, no reviews (the benchmark default)
    review    review every evaluated sub-agent answer and write a report per answer
    research  review + feed findings to the hypothesis ledger (multi_agent_research.py only)
"""
import os
from pathlib import Path

from agenticeoecosystem.definitions import ROOT_DIR

MODES = ("off", "review", "research")
MODE = "off"

# Sub-agents whose answers are recorded and reviewed: the ones described in
# rules.AGENT_CAPABILITIES. Never add hypothesis_agent (it would review its own bookkeeping).
EVALUATED_AGENTS = {
    "cdse_agent", "eumetsat_agent", "gee_agent", "climate_agent", "eo_stac_agent",
    "graph_rag_agent", "mooc_rag_agent", "arxiv_rag_agent", "gis_agent", "vlm_agent", "tle_agent",
}

# Share of answers grounded against their tool outputs in review mode. Grounding is the
# only step that yields a correctness score, so below 1.0 most reports have none.
# Research mode always grounds. Lower it to save reviewer tokens.
GROUNDING_SAMPLE_RATE = float(os.getenv("AGENTICEO_HELPER_GROUNDING_RATE", "1.0"))

# How long finish_task() waits for pending reviews after a task (outside the task timeout).
DRAIN_TIMEOUT_S = 300

WRITE_REPORTS = True      # one JSON report per reviewed answer, under run_dir(run_id)/reports
# Off: reviewer.get_evaluation_feedback() is not given to any agent as a tool yet, so
# reports queued for the coordinator would never be read and would pile up in memory.
FEED_COORDINATOR = False
ESCALATE = True           # currently only logs a warning (reviewer.notify_human)


def output_root() -> Path:
    """<output-dir>/helper next to the run's artifacts (as run_research.py documents),
    or output/helper when no run artifact root is set."""
    if os.getenv("EO_ARTIFACT_ROOT"):
        return Path(os.environ["EO_ARTIFACT_ROOT"]).resolve().parent / "helper"
    return ROOT_DIR / "output" / "helper"


def run_dir(run_id: str) -> Path:
    return output_root() / run_id


def feeds_hypotheses() -> bool:
    return MODE == "research"
