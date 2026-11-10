"""Hypothesis ledger: explicit, per-run state outside any model's context window.

Who writes what
- The evaluator only appends Findings (Ledger.add_findings).
- Only the hypothesis agent changes hypotheses, through the tool functions below.
- No LLM ever sets a confidence: scores are computed from recorded evidence.

How strong a piece of evidence is
    weight = diagnosticity (hypothesis agent: does this finding discriminate?)
           x reliability   (evaluation report: can this finding be trusted?)
Findings from the same report are not independent, so each report counts at most
once per direction (its strongest support and its strongest contradiction).
"""
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agenticeoecosystem.agents.helper.schemas import Finding

# ===================== POLICY: every number here is a DECISION =====================
DIAGNOSTICITY_WEIGHT = {"weak": 0.5, "moderate": 1.0, "strong": 2.0, "decisive": 4.0}
RELIABILITY_FACTOR = {"high": 1.0, "medium": 0.6, "low": 0.3}  # 'unverified' leads can't be evidence
STALE_EVIDENCE_FACTOR = 0.5   # evidence judged against an earlier version of the hypothesis
MAX_ACTIVE = 7                # forces merging/retiring instead of sprawl
SUPPORT_THRESHOLD = 4.0       # net score needed to mark 'supported'...
MIN_SUPPORTING_REPORTS = 2    # ...from at least this many separate reports
REFUTE_THRESHOLD = -4.0       # net score that allows 'refuted' without a decisive finding
CONTESTED_MIN = 2.0           # both sides at least this strong -> 'contested'
WAIT_FOR_EVALUATIONS_S = 120  # how long get_pending_findings waits for running evaluations
# TODO: independence is keyed on report_id. Two reports built on the same ERA5 field are
# still correlated; key on the source dataset/file once findings carry it.

Status = Literal["active", "supported", "refuted", "retired"]
Direction = Literal["supports", "contradicts", "neutral"]
Diagnosticity = Literal["weak", "moderate", "strong", "decisive"]
_STATUSES = {"active", "supported", "refuted", "retired"}
_DIRECTIONS = {"supports", "contradicts", "neutral"}
_DIAGNOSTICITY = set(DIAGNOSTICITY_WEIGHT)
_REVISION_KINDS = {"reword", "refine", "generalize"}


class Evidence(BaseModel):
    finding_id: str
    report_id: str
    finding_text: str
    reliability: str            # copied from the finding, never set by the agent
    direction: Direction
    diagnosticity: Diagnosticity
    rationale: str
    hypothesis_version: int
    at: float = Field(default_factory=time.time)


class Anomaly(BaseModel):
    finding_id: str
    note: str
    resolved_by: Optional[str] = None   # hypothesis id that now explains it


class Hypothesis(BaseModel):
    id: str
    statement: str
    scope: str                  # conditions: region, period, variables, scales; or 'general'
    predictions: list[str]      # observable consequences -> basis for discriminating questions
    falsification_test: str     # required: no test, no hypothesis
    status: Status = "active"
    version: int = 1            # bumped by refine/generalize
    evidence: dict[str, Evidence] = Field(default_factory=dict)  # keyed by finding_id
    parent_id: Optional[str] = None
    origin_finding_ids: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)  # every change, with reason
    created_at: float = Field(default_factory=time.time)


# =============================== Scoring ===============================

def evidence_weight(h: Hypothesis, e: Evidence) -> float:
    if e.direction == "neutral":
        return 0.0
    w = DIAGNOSTICITY_WEIGHT[e.diagnosticity] * RELIABILITY_FACTOR.get(e.reliability, 0.0)
    if e.hypothesis_version < h.version:
        w *= STALE_EVIDENCE_FACTOR
    return w if e.direction == "supports" else -w


def score(h: Hypothesis) -> dict:
    per_report = defaultdict(lambda: [0.0, 0.0])  # report_id -> [max support, max contradiction]
    for e in h.evidence.values():
        w = evidence_weight(h, e)
        if w > 0:
            per_report[e.report_id][0] = max(per_report[e.report_id][0], w)
        elif w < 0:
            per_report[e.report_id][1] = max(per_report[e.report_id][1], -w)
    pro = sum(p for p, _ in per_report.values())
    con = sum(c for _, c in per_report.values())
    net = pro - con
    decisive_against = any(
        e.direction == "contradicts" and e.diagnosticity == "decisive"
        and e.reliability in ("high", "medium") and e.hypothesis_version == h.version
        for e in h.evidence.values())

    if pro == 0 and con == 0:
        standing = "untested"
    elif pro >= CONTESTED_MIN and con >= CONTESTED_MIN:
        standing = "contested"
    elif net >= SUPPORT_THRESHOLD / 2:
        standing = "favoured"
    elif net <= REFUTE_THRESHOLD / 2:
        standing = "disfavoured"
    else:
        standing = "inconclusive"

    return {"pro": round(pro, 2), "con": round(con, 2), "net": round(net, 2),
            "supporting_reports": sum(1 for p, _ in per_report.values() if p > 0),
            "decisive_against": decisive_against, "standing": standing}


# =============================== Ledger ===============================

class Ledger:
    """Per-run. The JSON file stays on disk after the run, for audit."""

    def __init__(self):
        self.run_id: Optional[str] = None
        self.path: Optional[Path] = None
        self.hypotheses: dict[str, Hypothesis] = {}
        self.findings: dict[str, Finding] = {}    # every finding this run, processed or not
        self.pending: list[str] = []              # finding ids not yet processed
        self.anomalies: dict[str, Anomaly] = {}   # keyed by finding id

    def reset(self, run_id: str, run_dir: Path) -> None:
        self.run_id, self.path = run_id, Path(run_dir) / "hypotheses.json"
        self.hypotheses, self.findings, self.pending, self.anomalies = {}, {}, [], {}
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "run_id": self.run_id,
            "hypotheses": [h.model_dump() | {"score": score(h)} for h in self.hypotheses.values()],
            "findings": [f.model_dump() for f in self.findings.values()],
            "pending": self.pending,
            "anomalies": [a.model_dump() for a in self.anomalies.values()],
        }, indent=1, default=str))

    def add_findings(self, findings: list[Finding], run_id: Optional[str]) -> None:
        if run_id != self.run_id:
            return  # late evaluation from a previous run
        for f in findings:
            self.findings[f.finding_id] = f
            self.pending.append(f.finding_id)
        if findings:
            self.save()

    def live(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == "active"]

    def resolve_anomalies(self, finding_ids: list[str], by: str) -> list[str]:
        resolved = []
        for fid in finding_ids:
            a = self.anomalies.get(fid)
            if a and a.resolved_by is None:
                a.resolved_by = by
                resolved.append(fid)
        return resolved


ledger = Ledger()


# ============ Tools for the hypothesis agent (plain functions -> FunctionTool) ============
# No default arguments: some ADK/model combinations handle them badly.

def _get(hypothesis_id: str):
    h = ledger.hypotheses.get(hypothesis_id)
    return (h, None) if h else (None, {"error": f"unknown hypothesis {hypothesis_id}"})


def get_ledger() -> dict:
    """Returns all hypotheses with their computed score and standing, the evidence already
    recorded for each, and open anomalies (findings no live hypothesis explains). Call this first."""
    return {
        "hypotheses": [{
            "id": h.id, "statement": h.statement, "scope": h.scope, "predictions": h.predictions,
            "falsification_test": h.falsification_test, "status": h.status, "version": h.version,
            "parent_id": h.parent_id, "score": score(h),
            "evidence": [{"finding_id": e.finding_id, "direction": e.direction,
                          "diagnosticity": e.diagnosticity, "reliability": e.reliability,
                          "stale": e.hypothesis_version < h.version}
                         for e in h.evidence.values()],
        } for h in ledger.hypotheses.values()],
        "open_anomalies": [{"finding_id": a.finding_id, "note": a.note,
                            "text": ledger.findings[a.finding_id].text}
                           for a in ledger.anomalies.values() if a.resolved_by is None],
        "limits": {"max_active": MAX_ACTIVE, "active": len(ledger.live())},
    }


async def get_pending_findings() -> dict:
    """Returns new findings from evaluated subagent answers. Each finding has a kind
    (result, unexpected, lead) and a reliability computed from its evaluation report.
    Waits briefly for evaluations that are still running."""
    from agenticeoecosystem.agents.helper.reviewer import drain, pending_count
    await drain(timeout=WAIT_FOR_EVALUATIONS_S)
    keep = {"finding_id", "kind", "reliability", "reliability_notes", "text",
            "agent_name", "question", "source_tool"}
    return {"findings": [ledger.findings[i].model_dump(include=keep) for i in ledger.pending],
            "evaluations_still_running": pending_count()}


def propose_hypothesis(statement: str, scope: str, predictions: list[str], falsification_test: str,
                       rationale: str, origin_finding_ids: list[str], parent_id: str) -> dict:
    """Creates a new active hypothesis.
    scope: the conditions it applies to (region, period, variables), or 'general'.
    predictions: observable consequences that a subagent could check.
    falsification_test: a concrete observation that would refute it.
    origin_finding_ids: findings or anomalies that motivated it ([] if none); anomalies listed are resolved.
    parent_id: id of the hypothesis this splits from or specialises, or '' if none."""
    if len(ledger.live()) >= MAX_ACTIVE:
        return {"error": f"{MAX_ACTIVE} active hypotheses already. Retire or merge one first."}
    if not falsification_test.strip():
        return {"error": "A falsification test is required."}
    if not predictions:
        return {"error": "At least one observable prediction is required."}
    unknown = [i for i in origin_finding_ids if i not in ledger.findings]
    if unknown:
        return {"error": f"unknown finding ids: {unknown}"}
    if parent_id and parent_id not in ledger.hypotheses:
        return {"error": f"unknown parent {parent_id}"}

    h = Hypothesis(id="H" + uuid.uuid4().hex[:6], statement=statement, scope=scope or "general",
                   predictions=predictions, falsification_test=falsification_test,
                   parent_id=parent_id or None, origin_finding_ids=origin_finding_ids,
                   history=[f"created: {rationale}"])
    ledger.hypotheses[h.id] = h
    resolved = ledger.resolve_anomalies(origin_finding_ids, h.id)
    ledger.save()
    return {"created": h.id, "anomalies_resolved": resolved}


def record_evidence(hypothesis_id: str, finding_id: str, direction: str,
                    diagnosticity: str, rationale: str) -> dict:
    """Records how a finding bears on a hypothesis. Recording the same finding again for the
    same hypothesis replaces the earlier assessment.
    direction: supports | contradicts | neutral.
    diagnosticity: how much more likely the finding is if the hypothesis is true than if it is
    false (for contradicts, the reverse): weak | moderate | strong | decisive.
    'decisive' means the finding meets, or directly fails, the falsification test.
    Reliability comes from the evaluation report; you cannot set it."""
    h, err = _get(hypothesis_id)
    if err:
        return err
    if h.status == "retired":
        return {"error": f"{hypothesis_id} is retired."}
    f = ledger.findings.get(finding_id)
    if not f:
        return {"error": f"unknown finding {finding_id}"}
    if f.kind == "lead":
        return {"error": "Leads are not backed by tool outputs and cannot count as evidence. "
                         "Turn them into a NEXT QUESTION, or record_anomaly if relevant."}
    if direction not in _DIRECTIONS:
        return {"error": f"direction must be one of {sorted(_DIRECTIONS)}"}
    if diagnosticity not in _DIAGNOSTICITY:
        return {"error": f"diagnosticity must be one of {sorted(_DIAGNOSTICITY)}"}

    e = Evidence(finding_id=f.finding_id, report_id=f.report_id, finding_text=f.text,
                 reliability=f.reliability, direction=direction, diagnosticity=diagnosticity,
                 rationale=rationale, hypothesis_version=h.version)
    h.evidence[finding_id] = e
    h.history.append(f"v{h.version} evidence {finding_id}: {direction}/{diagnosticity} "
                     f"(reliability {f.reliability}): {rationale}")
    if direction == "supports":
        ledger.resolve_anomalies([finding_id], h.id)
    ledger.save()
    return {"ok": True, "weight": round(evidence_weight(h, e), 2), "score_now": score(h)}


def record_anomaly(finding_id: str, note: str) -> dict:
    """Marks a finding that is relevant to the investigation but explained by no live hypothesis.
    Open anomalies stay visible in get_ledger() until a hypothesis is created or revised to explain them."""
    if finding_id not in ledger.findings:
        return {"error": f"unknown finding {finding_id}"}
    ledger.anomalies[finding_id] = Anomaly(finding_id=finding_id, note=note)
    ledger.save()
    return {"ok": True, "open_anomalies": sum(a.resolved_by is None for a in ledger.anomalies.values())}


def revise_hypothesis(hypothesis_id: str, kind: str, new_statement: str, new_scope: str,
                      new_predictions: list[str], new_falsification_test: str,
                      reason: str, triggering_finding_ids: list[str]) -> dict:
    """Changes a hypothesis. Pass '' or [] for fields you are not changing.
    kind: reword     -> same claim, clearer wording; evidence keeps full weight
          refine     -> narrower scope or more specific claim
          generalize -> broader scope
    For refine/generalize the version increases and earlier evidence counts at reduced weight
    until you re-assess it with record_evidence.
    Anomalies listed in triggering_finding_ids are resolved by this revision."""
    h, err = _get(hypothesis_id)
    if err:
        return err
    if kind not in _REVISION_KINDS:
        return {"error": f"kind must be one of {sorted(_REVISION_KINDS)}"}
    if not reason.strip():
        return {"error": "A reason is required."}

    h.history.append(f"v{h.version} {kind}: '{h.statement}' [scope: {h.scope}] "
                     f"triggered by {triggering_finding_ids}: {reason}")
    if new_statement.strip():
        h.statement = new_statement
    if new_scope.strip():
        h.scope = new_scope
    if new_predictions:
        h.predictions = new_predictions
    if new_falsification_test.strip():
        h.falsification_test = new_falsification_test
    if kind != "reword":
        h.version += 1
    resolved = ledger.resolve_anomalies(triggering_finding_ids, h.id)
    ledger.save()
    stale = [fid for fid, e in h.evidence.items() if e.hypothesis_version < h.version]
    return {"ok": True, "version": h.version, "evidence_to_reassess": stale,
            "anomalies_resolved": resolved}


def set_status(hypothesis_id: str, status: str, reason: str) -> dict:
    """Sets status: active | supported | refuted | retired. A reason is required.
    'supported' and 'refuted' are checked against the recorded evidence and may be refused.
    Use 'retired' for hypotheses that are superseded, merged into another, or no longer relevant."""
    h, err = _get(hypothesis_id)
    if err:
        return err
    if status not in _STATUSES:
        return {"error": f"status must be one of {sorted(_STATUSES)}"}
    if not reason.strip():
        return {"error": "A reason is required."}
    s = score(h)
    if status == "supported" and not (s["net"] >= SUPPORT_THRESHOLD
                                      and s["supporting_reports"] >= MIN_SUPPORTING_REPORTS):
        return {"error": f"Refused: needs net >= {SUPPORT_THRESHOLD} from >= {MIN_SUPPORTING_REPORTS} "
                         f"reports; current {s}. Ask a question that would settle it."}
    if status == "refuted" and not (s["decisive_against"] or s["net"] <= REFUTE_THRESHOLD):
        return {"error": f"Refused: needs a decisive contradiction or net <= {REFUTE_THRESHOLD}; "
                         f"current {s}."}
    if status == "active" and h.status != "active" and len(ledger.live()) >= MAX_ACTIVE:
        return {"error": f"{MAX_ACTIVE} active hypotheses already."}

    h.history.append(f"status {h.status} -> {status}: {reason}")
    h.status = status
    ledger.save()
    return {"ok": True, "score": s}


def mark_findings_processed(finding_ids: list[str]) -> dict:
    """Removes findings from the pending queue. Include findings you skipped as irrelevant."""
    done = set(finding_ids)
    ledger.pending = [i for i in ledger.pending if i not in done]
    ledger.save()
    return {"remaining": len(ledger.pending)}


# ===================== Read-only tool for the coordinator =====================

def get_hypothesis_summary() -> dict:
    """Read-only summary of the current hypotheses: statement, scope, status, standing and
    falsification test. Use it to plan questions or write the final answer."""
    return {"hypotheses": [{"id": h.id, "statement": h.statement, "scope": h.scope,
                            "status": h.status, "standing": score(h)["standing"],
                            "falsification_test": h.falsification_test}
                           for h in ledger.hypotheses.values()],
            "open_anomalies": sum(a.resolved_by is None for a in ledger.anomalies.values())}