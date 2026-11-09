import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------- Stage 1: blind checklist (never sees the answer) ----------------

class ChecklistItem(BaseModel):
    id: str
    description: str
    importance: Literal["essential", "useful"]   # DECISION: two tiers


class Checklist(BaseModel):
    items: list[ChecklistItem]
    question_issues: list[str] = Field(
        default_factory=list,
        description="Ambiguities or missing context in the QUESTION itself.")


# ---------------- Stage 2: grading ----------------

class ItemVerdict(BaseModel):
    item_id: str
    present: Literal["yes", "partial", "no"]
    lines: list[int] = Field(default_factory=list, description="Answer line numbers containing it.")


class LineLabel(BaseModel):
    line: int
    relevance: Literal["unexpected_but_valuable", "noise"]
    note: str = ""


class Grading(BaseModel):
    verdicts: list[ItemVerdict]
    extra_valid_items: list[str] = Field(default_factory=list)
    off_topic_lines: list[LineLabel] = Field(
        default_factory=list,
        description="ONLY lines that are not on topic. Unlisted lines are on topic.")


# ---------------- Stage 3: grounding ----------------

class ClaimCheck(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "contradicted", "not_checkable"]
    claim_type: Literal["result", "provenance"] = Field(
        "result",
        description="result = says something about the world or the data (a value, trend, pattern, "
                    "comparison); provenance = what was downloaded, processed or saved.")
    lines: list[int] = Field(default_factory=list,
                             description="Answer line numbers where the claim appears.")
    trace_step: Optional[int] = None
    note: str = ""


class Grounding(BaseModel):
    claims: list[ClaimCheck]


# ---------------- Findings: the evaluator -> hypothesis ledger interface ----------------

Reliability = Literal["high", "medium", "low", "unverified"]


class Finding(BaseModel):
    """Something the evaluator noticed. It never says which hypothesis it bears on:
    the hypothesis agent decides that."""
    finding_id: str
    report_id: str
    run_id: Optional[str]
    agent_name: str
    question: str                 # what the subagent was asked
    text: str
    kind: Literal["result", "unexpected", "lead"]
    reliability: Reliability      # computed from the report, never by an LLM
    reliability_notes: list[str] = Field(default_factory=list)
    trace_step: Optional[int] = None
    source_tool: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


# ---------------- Final report ----------------

class EvaluationReport(BaseModel):
    report_id: str
    run_id: Optional[str]
    agent_name: str
    question: str
    question_source: str
    answer: str
    coverage: Optional[int] = None
    focus: Optional[int] = None
    correctness: Optional[int] = None   # None = grounding not run
    checklist: Optional[Checklist] = None
    grading: Optional[Grading] = None
    grounding: Optional[Grounding] = None
    rule_violations: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    escalate: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)
    created_at: float
    reviewer_tokens: dict = Field(default_factory=dict)