"""
Virtual Lab — structured scientific deliberation over the existing EO sub-agents.

Optional alongside the `multi` orchestrator: nothing here is imported by
agents/multi_agent.py. Select it with pipeline="virtual_lab" in
evaluation/runners/pipeline_runner.py or call `run_virtual_lab` directly.
Importing this package is side-effect free; sub-agents load only when a lab
is built.
"""

from .config import VirtualLabConfig
from .schemas import (
    DecisionStatus,
    EvidenceItem,
    EvidenceKind,
    ExperimentPlan,
    ExperimentResult,
    Hypothesis,
    LabRecord,
    ScientificCritique,
    ScientificDecision,
    ScientificProblem,
    ScientificProposal,
)

__all__ = [
    "VirtualLabConfig", "DecisionStatus", "EvidenceItem", "EvidenceKind", "ExperimentPlan",
    "ExperimentResult", "Hypothesis", "LabRecord", "ScientificCritique", "ScientificDecision",
    "ScientificProblem", "ScientificProposal",
]
