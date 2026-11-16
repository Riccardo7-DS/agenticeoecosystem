"""
schemas.py — Validated scientific artifacts exchanged during a Virtual Lab run.

Every artifact an LLM participant emits is parsed into one of these models
before it is written to session state, so downstream phases never reason over
free text that failed validation. Evidence is typed (retrieved / assumption /
hypothesis / verified) so agent agreement can never be mistaken for proof: only
an ExperimentResult can back a VERIFIED claim.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceKind(StrEnum):
    RETRIEVED = "retrieved"      # returned by a tool call made during this run
    ASSUMPTION = "assumption"    # stated by an agent, not checked
    HYPOTHESIS = "hypothesis"    # a testable claim, not yet tested
    VERIFIED = "verified"        # backed by an ExperimentResult


class EvidenceItem(BaseModel):
    """A single claim plus where it came from."""

    claim: str
    kind: EvidenceKind
    source: str | None = Field(
        default=None,
        description="Tool/sub-agent name, dataset, or citation the claim came from.",
    )
    tool_call_ids: list[str] = Field(
        default_factory=list,
        description="Ledger ids of the tool calls backing a RETRIEVED claim (filled by the coordinator).",
    )
    result_id: str | None = Field(
        default=None, description="ExperimentResult id backing a VERIFIED claim."
    )

    @model_validator(mode="after")
    def _verified_needs_result(self) -> EvidenceItem:
        if self.kind == EvidenceKind.VERIFIED and not self.result_id:
            raise ValueError("a VERIFIED claim must reference an ExperimentResult (result_id)")
        return self


class StudyArea(BaseModel):
    name: str
    bbox: list[float] | None = Field(
        default=None, description="[west, south, east, north] in WGS-84 degrees."
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox must be [west, south, east, north]")
        west, south, east, north = v
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox longitudes must be within [-180, 180]")
        if not (-90 <= south < north <= 90):
            raise ValueError("bbox latitudes must satisfy -90 <= south < north <= 90")
        return v


class TimePeriod(BaseModel):
    start: str = Field(description="YYYY-MM-DD")
    end: str = Field(description="YYYY-MM-DD")


TaskType = Literal[
    "dataset_selection", "scientific_reasoning", "hypothesis_testing", "experimental_execution"
]


class ScientificProblem(BaseModel):
    question: str
    objectives: list[str]
    study_area: StudyArea
    time_period: TimePeriod
    constraints: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    data_needs: list[str] = Field(
        default_factory=list, description="2-6 catalogue-style descriptions of the data the "
                                          "problem needs (instrument/technique, product form, "
                                          "quantity, scale); the lab's data discovery searches "
                                          "every catalogue with them."
    )
    task_types: list[TaskType] = Field(default_factory=list)
    participants: list[str] = Field(
        default_factory=list, description="Specialist names selected for this problem."
    )
    sub_agents: list[str] = Field(
        default_factory=list, description="Existing sub-agents the lab may call for this problem "
                                          "(empty: all offered)."
    )
    rationale: str = ""


class Hypothesis(BaseModel):
    id: str
    statement: str
    prediction: str = Field(description="Observable consequence if the hypothesis holds.")
    falsification: str = Field(description="Result that would refute the hypothesis.")
    status: Literal["untested", "supported", "refuted", "inconclusive"] = "untested"


# Role words models use outside the four roles, mapped to the closest one. A reference or
# benchmark validates; training data is what the method runs on; a comparison is a baseline.
_DATASET_ROLE_SYNONYMS = {
    "validation": "reference", "benchmark": "reference", "ground_truth": "reference",
    "ground truth": "reference", "test": "reference", "evaluation": "reference",
    "training": "primary", "pretraining": "primary", "input": "primary", "main": "primary",
    "comparison": "baseline", "control": "baseline",
    "auxiliary": "ancillary", "supporting": "ancillary", "context": "ancillary",
}


_DATASET_DECISION_SYNONYMS = {
    "use": "selected", "used": "selected", "select": "selected", "chosen": "selected",
    "reject": "rejected", "not_selected": "rejected", "not selected": "rejected",
    "excluded": "rejected", "discarded": "rejected",
}
# Registry ids: a ledger call id, optionally "/<collection id>" for one collection of a
# search_catalogues result (e.g. "tc-0052/NISAR_L2_GSLC_V1").
_REGISTRY_ID = re.compile(r"tc-\d{4}(?:/[^\s,;\]\)\"']+)?")


class DatasetChoice(BaseModel):
    name: str = Field(description="Sensor/product name, ideally a sensor catalog key.")
    role: Literal["primary", "reference", "baseline", "ancillary"] = "primary"
    observable: str = Field(default="", description="What this dataset is used to observe.")
    justification: str = ""
    decision: Literal["selected", "rejected", "unverified"] = Field(
        default="selected",
        description="selected: a confirmed input; rejected: considered and not used (say why in "
                    "justification); unverified: a possible option not shown to exist here.")
    source_ids: list[str] = Field(
        default_factory=list,
        description="Dataset-registry ids (DATA INVENTORY rows or catalogue tool results of this "
                    "run) showing the data exist for the study area and period.")

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, v):
        if isinstance(v, str):
            key = v.strip().lower()
            key = _DATASET_DECISION_SYNONYMS.get(key, key)
            # An unrecognised word must not turn into a confirmed input.
            return key if key in ("selected", "rejected", "unverified") else "unverified"
        return v

    @field_validator("source_ids", mode="before")
    @classmethod
    def _normalise_source_ids(cls, v):
        """Accept a string or list, with or without an "inventory:" prefix."""
        if v is None:
            return []
        items = [v] if isinstance(v, str) else v if isinstance(v, list) else []
        return list(dict.fromkeys(m for item in items if isinstance(item, str)
                                  for m in _REGISTRY_ID.findall(item)))

    @property
    def in_use(self) -> bool:
        """Selected or unverified: part of the approach (rejected ones are not)."""
        return self.decision != "rejected"

    @field_validator("role", mode="before")
    @classmethod
    def _normalise_role(cls, v):
        if isinstance(v, str):
            key = v.strip().lower()
            return _DATASET_ROLE_SYNONYMS.get(key, key)
        return v


class ScientificProposal(BaseModel):
    agent: str
    summary: str
    hypotheses: list[Hypothesis]
    datasets: list[DatasetChoice]
    methods: list[str]
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    revision: int = 0
    changes_from_previous: str | None = None


IssueCategory = Literal[
    "unsupported_claim",
    "physical_inconsistency",
    "dataset_availability",
    "dataset_suitability",
    "coverage",
    "methodology",
    "evidence_sufficiency",
    "limitation",
    "other",
]
Severity = Literal["blocking", "major", "minor"]


class CritiqueIssue(BaseModel):
    category: IssueCategory
    severity: Severity
    claim: str = Field(description="The specific claim or choice being challenged.")
    rationale: str
    suggested_fix: str = ""


class ScientificCritique(BaseModel):
    author: str
    target: str = Field(description="Agent whose proposal is critiqued.")
    issues: list[CritiqueIssue] = Field(default_factory=list)
    round: int = 0

    @property
    def has_blocking(self) -> bool:
        return any(i.severity == "blocking" for i in self.issues)


class CritiqueBatch(BaseModel):
    """Wrapper so one LLM call can critique several proposals."""

    critiques: list[ScientificCritique]


class ExperimentStep(BaseModel):
    order: int
    description: str
    agent: str = Field(description="Existing sub-agent that would execute this step.")
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list,
                                  description="Ids of the hypotheses this step helps test.")


class ExperimentPlan(BaseModel):
    research_question: str
    hypotheses: list[Hypothesis]
    datasets: list[DatasetChoice]
    workflow: list[ExperimentStep]
    reference_data: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    statistical_tests: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    stopping_criteria: list[str] = Field(default_factory=list)
    analyses: list[str] = Field(default_factory=list,
                                description="Execute mode: the lab's fixed analyses to run (analyses.ANALYSES).")


class ExperimentResult(BaseModel):
    """Outcome of one executed analysis. The analysis reports numbers; which hypothesis they
    support is judged afterwards by the PI's conclusion."""

    id: str
    analysis: str = ""
    status: Literal["ok", "failed"] = "ok"
    summary: str = ""
    hypothesis_id: str = ""
    outcome: Literal["supports", "refutes", "inconclusive"] = "inconclusive"
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class DecisionStatus(StrEnum):
    DECIDED = "decided"                              # a defensible workflow/decision was reached
    NEEDS_EXPERIMENT = "needs_experiment"            # competing hypotheses need testing first
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # no reliable conclusion possible
    BUDGET_EXHAUSTED = "budget_exhausted"            # stopped by a resource bound


class Disagreement(BaseModel):
    topic: str
    positions: dict[str, str] = Field(description="agent name -> position")
    evidence_needed: str = ""


class ProvenanceLink(BaseModel):
    source: str = Field(description="Id of the supporting item (proposal, critique, tool call, result).")
    relation: Literal[
        "based_on_proposal", "addresses_critique", "cites_tool_call", "supported_by_result"
    ]


class ScientificDecision(BaseModel):
    status: DecisionStatus
    summary: str
    recommended_workflow: list[str] = Field(default_factory=list)
    selected_hypotheses: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    justification: list[EvidenceItem] = Field(default_factory=list)
    provenance: list[ProvenanceLink] = Field(default_factory=list)
    status_note: str = Field(
        default="", description="Why the coordinator changed the status the PI proposed, if it did."
    )

    @property
    def claims_verified(self) -> bool:
        """True only if at least one justification is backed by an experiment result."""
        return any(e.kind == EvidenceKind.VERIFIED for e in self.justification)


class PaperRef(BaseModel):
    arxiv_id: str
    title: str = ""
    year: int | None = None
    citation_count: int | None = Field(default=None, description="Semantic Scholar; null if unknown.")


LiteratureVerdict = Literal["supports", "contradicts", "mixed", "inconclusive", "not_searched"]


class LiteratureCheck(BaseModel):
    """A contested claim checked against arXiv papers, ranked by citations."""

    claim_id: str
    claim: str
    objection: str = Field(default="", description="Why it was contested.")
    raised_by: str = Field(description="Agent that contested the claim.")
    target: str | None = Field(default=None, description="Agent whose claim is contested.")
    round: int = 0
    scope: Literal["general", "event_specific"] = "general"
    queries: list[str] = Field(default_factory=list)
    papers: list[PaperRef] = Field(default_factory=list)
    verdict: LiteratureVerdict = "not_searched"
    summary: str = ""
    tool_call_ids: list[str] = Field(default_factory=list)


class LiteratureQuery(BaseModel):
    claim_id: str
    scope: Literal["general", "event_specific"] = "general"
    queries: list[str] = Field(default_factory=list)


class LiteratureQueryBatch(BaseModel):
    """literature_reviewer output, step 1: how to search each contested claim."""

    claims: list[LiteratureQuery] = Field(default_factory=list)


class LiteratureAssessment(BaseModel):
    claim_id: str
    verdict: Literal["supports", "contradicts", "mixed", "inconclusive"]
    summary: str = ""
    papers: list[str] = Field(default_factory=list, description="arXiv ids the verdict rests on.")


class LiteratureAssessmentBatch(BaseModel):
    """literature_reviewer output, step 2: what the retrieved papers say."""

    assessments: list[LiteratureAssessment] = Field(default_factory=list)


class LabRecord(BaseModel):
    """Everything one lab run produced; persisted as an external artifact."""

    question: str
    config: dict
    problem: ScientificProblem | None = None
    proposal_history: list[ScientificProposal] = Field(default_factory=list)
    critiques: list[ScientificCritique] = Field(default_factory=list)
    final_proposals: dict[str, ScientificProposal] = Field(default_factory=dict)
    decision: ScientificDecision | None = None
    plan: ExperimentPlan | None = None
    plan_validation: ScientificCritique | None = None
    results: list[ExperimentResult] = Field(default_factory=list)
    literature: list[LiteratureCheck] = Field(default_factory=list)
    termination: str = ""
    rounds_run: int = 0
    invalid_outputs: list[dict] = Field(default_factory=list)
    data_grounding: dict[str, Any] = Field(
        default_factory=dict, description="Dataset-grounding metrics (validation.grounding_metrics).")
