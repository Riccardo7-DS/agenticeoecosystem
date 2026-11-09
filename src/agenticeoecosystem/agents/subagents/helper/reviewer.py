"""Reviewer: a fixed pipeline run in the background after each evaluated sub-agent answer.

Stage 0  deterministic per-agent rules (free, always)
Stage 1  blind checklist from the question only
Stage 2  grade the answer against the checklist -> coverage, focus
Stage 3  ground claims against tool outputs -> correctness, findings
Stage 4  scores, escalation, dispatch (report files, ledger, humans)
"""
import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agenticeoecosystem.agents.helper import config
from agenticeoecosystem.agents.helper.ledger import ledger
from agenticeoecosystem.agents.helper.llm_call import run_structured
from agenticeoecosystem.agents.helper.rules import AGENT_CAPABILITIES, RULES
from agenticeoecosystem.agents.helper.schemas import (
    Checklist, EvaluationReport, Finding, Grading, Grounding,
)
from agenticeoecosystem.agents.helper.trace import Interaction, numbered_lines

logger = logging.getLogger(__name__)


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"[{i}] {l}" for i, l in enumerate(lines))

# ========================= Stage 1: blind checklist =========================
CHECKLIST_INSTRUCTION = """You define what a complete answer to a question must contain.
You will NOT see the answer. List the content items a competent answer must include.
Mark an item 'essential' only if the answer would be wrong or useless without it.
For data/analysis requests include procedural items (aggregation method, units, region, period).
Keep the list short: typically 3-8 items. Do not pad.
Separately, list any ambiguity or missing context in the question itself."""
# DECISION: the 3-8 item range strongly shapes coverage scores. Tune it.


async def build_checklist(it: Interaction, rq: str, usage: dict) -> Optional[Checklist]:
    prompt = (f"Overall research question (context only): {rq or 'unknown'}\n"
              f"Agent capabilities: {AGENT_CAPABILITIES.get(it.agent_name, 'unknown')}\n\n"
              f"Question asked to the agent:\n{it.question}")
    return await run_structured(CHECKLIST_INSTRUCTION, prompt, Checklist, usage=usage)

# =========================== Stage 2: grading ===========================
GRADING_INSTRUCTION = """You compare an answer to a checklist. The answer is split into numbered lines.
For each checklist item: is it in the answer (yes/partial/no), and on which line numbers?
List valid content the answer includes that the checklist missed.
List ONLY the lines that are not on topic, labelled:
- unexpected_but_valuable: not asked for, but plausibly important for the overall research question
- noise: irrelevant filler
Unlisted lines count as on topic. Judge presence and relevance only, NOT correctness."""


async def grade(it: Interaction, checklist: Checklist, rq: str, lines: list[str],
                usage: dict) -> Optional[Grading]:
    prompt = (f"Overall research question: {rq or 'unknown'}\n"
              f"Agent capabilities: {AGENT_CAPABILITIES.get(it.agent_name, 'unknown')}\n\n"
              f"Question:\n{it.question}\n\nChecklist:\n{checklist.model_dump_json(indent=1)}\n\n"
              f"Answer (numbered lines):\n{_numbered(lines)}")
    return await run_structured(GRADING_INSTRUCTION, prompt, Grading, usage=usage)

# ========================== Stage 3: grounding ==========================
GROUNDING_INSTRUCTION = """You check the factual claims in an answer against the tool outputs that produced it.
The answer is split into numbered lines.
Extract the substantive claims (numbers, units, dates, variables, regions, file paths, conclusions).
For each claim:
- give the answer line numbers where it appears
- claim_type: 'result' if it says something about the world or the data (a value, trend, pattern,
  comparison); 'provenance' if it says what was downloaded, processed or saved
- cite the tool step that supports or contradicts it, and set status:
  supported: a step backs it
  contradicted: a step says otherwise
  unsupported: no step backs it
  not_checkable: a judgement or plan, not a factual claim
Use only the tool steps provided. Ignore anything the answer says about its own reasoning."""


async def ground(it: Interaction, lines: list[str], usage: dict) -> Optional[Grounding]:
    if not it.steps:
        return None
    prompt = f"Answer (numbered lines):\n{_numbered(lines)}\n\nTool steps:\n{it.trace_text()}"
    return await run_structured(GROUNDING_INSTRUCTION, prompt, Grounding, usage=usage)


def should_ground(it: Interaction, grading: Optional[Grading], rule_violations: list[str]) -> bool:
    if not it.steps:
        return False
    if config.feeds_hypotheses():
        return True   # findings come only from grounded claims; sampling would starve the ledger
    if rule_violations:
        return True
    if grading and any(l.relevance == "unexpected_but_valuable" for l in grading.off_topic_lines):
        return True
    return random.random() < config.GROUNDING_SAMPLE_RATE

# ========================== Stage 4: scoring ==========================
IMPORTANCE_WEIGHT = {"essential": 2.0, "useful": 1.0}       # DECISION
PRESENCE_CREDIT = {"yes": 1.0, "partial": 0.5, "no": 0.0}   # DECISION


def to_likert(x: float) -> int:
    return 1 + round(4 * max(0.0, min(1.0, x)))


def coverage_score(checklist: Checklist, grading: Grading) -> int:
    by_id = {v.item_id: v for v in grading.verdicts}
    total = got = 0.0
    for item in checklist.items:
        w = IMPORTANCE_WEIGHT[item.importance]
        total += w
        if item.id in by_id:
            got += w * PRESENCE_CREDIT[by_id[item.id].present]
    score = to_likert(got / total) if total else 3
    # DECISION: a missing essential item caps coverage ("major omissions" = 2).
    if any(i.importance == "essential" and (i.id not in by_id or by_id[i.id].present == "no")
           for i in checklist.items):
        score = min(score, 2)
    return score


def focus_score(lines: list[str], grading: Grading) -> int:
    total = sum(len(l) for l in lines)
    if not total:
        return 3
    # Unexpected-but-valuable lines are deliberately NOT counted as off-focus.
    noise_idx = {l.line for l in grading.off_topic_lines
                 if l.relevance == "noise" and 0 <= l.line < len(lines)}
    return to_likert(1 - sum(len(lines[i]) for i in noise_idx) / total)


def correctness_score(grounding: Grounding, rule_violations: list[str]) -> int:
    checkable = [c for c in grounding.claims if c.status != "not_checkable"]
    if not checkable:
        return 2 if rule_violations else 3
    score = to_likert(sum(c.status == "supported" for c in checkable) / len(checkable))
    if any(c.status == "contradicted" for c in checkable):
        score = min(score, 2)   # DECISION
    if rule_violations:
        score = min(score, 3)   # DECISION
    return score

# ============================ Escalation ============================

def escalation(report: EvaluationReport) -> list[str]:
    reasons = []
    if report.grounding and any(c.status == "contradicted" for c in report.grounding.claims):
        reasons.append("answer contradicts its own tool outputs")
    if report.rule_violations and report.correctness is not None and report.correctness <= 2:
        reasons.append("procedural rule violated and low correctness")
    return reasons  # DECISION: this is where alarm fatigue is won or lost


def notify_human(report: EvaluationReport) -> None:
    # DECISION: Slack, email, a UI queue, or pausing the run.
    logger.warning("ESCALATION %s (%s): %s", report.report_id, report.agent_name,
                   report.escalation_reasons)

# ===================== Findings for the hypothesis ledger =====================
_LEVELS = ["low", "medium", "high"]


def _downgrade(level: str) -> str:
    return _LEVELS[max(0, _LEVELS.index(level) - 1)]


def report_reliability(report: EvaluationReport) -> tuple[str, list[str]]:
    """Reliability of the grounded content in one report. DECISION: these penalties."""
    level, notes = "high", []
    if report.rule_violations:
        level = _downgrade(level)
        notes.append(f"{len(report.rule_violations)} procedural issue(s) in this answer")
    if report.grounding and any(c.status == "contradicted" for c in report.grounding.claims):
        level = _downgrade(level)
        notes.append("the same answer contains claims contradicted by its own tool outputs")
    if report.coverage is not None and report.coverage <= 2:
        notes.append("answer missed essential content")   # noted, not penalised
    if report.question_source != "task_tool_args":
        notes.append("question captured via fallback; may not be what the coordinator asked")
    return level, notes


def extract_findings(report: EvaluationReport, lines: list[str], it: Interaction) -> list[Finding]:
    """DECISION: only grounded 'result' claims become evidence-grade findings.
    Provenance claims (files, downloads) never reach the ledger."""
    if report.grounding is None:
        return []
    level, notes = report_reliability(report)
    tools_by_step = {s.index: s.tool for s in it.steps}

    def make(text: str, kind: str, reliability: str, step: Optional[int] = None,
             extra_notes: tuple = ()) -> Finding:
        return Finding(finding_id="F" + uuid.uuid4().hex[:8], report_id=report.report_id,
                       run_id=report.run_id, agent_name=report.agent_name,
                       question=report.question[:500], text=text, kind=kind,
                       reliability=reliability, reliability_notes=notes + list(extra_notes),
                       trace_step=step, source_tool=tools_by_step.get(step))

    unexpected = ({l.line for l in report.grading.off_topic_lines
                   if l.relevance == "unexpected_but_valuable"} if report.grading else set())
    out: list[Finding] = []
    claims_by_line = defaultdict(list)

    for c in report.grounding.claims:
        for ln in c.lines:
            claims_by_line[ln].append(c)
        # Claims on unexpected lines are represented by the 'unexpected' finding below.
        if c.status == "supported" and c.claim_type == "result" and not set(c.lines) & unexpected:
            out.append(make(c.claim, "result", level, c.trace_step))

    for ln in sorted(unexpected):
        if not 0 <= ln < len(lines):
            continue
        cs = [c for c in claims_by_line.get(ln, []) if c.status != "not_checkable"]
        if any(c.status == "contradicted" for c in cs):
            continue   # DECISION: contradicted unexpected content is dropped, not even a lead
        if cs and all(c.status == "supported" for c in cs):
            out.append(make(lines[ln], "unexpected", level, cs[0].trace_step))
        else:
            out.append(make(lines[ln], "lead", "unverified",
                            extra_notes=("not backed by any tool output",)))
    return out

# ============================== Evaluate ==============================

async def evaluate(it: Interaction, rq: str, run_id: Optional[str]) -> EvaluationReport:
    usage = {"input": 0, "output": 0, "calls": 0}
    report = EvaluationReport(
        report_id=uuid.uuid4().hex[:12], run_id=run_id, agent_name=it.agent_name,
        question=it.question or "", question_source=it.question_source,
        answer=it.answer or "", created_at=time.time(),
    )
    for rule in RULES.get(it.agent_name, []):
        report.rule_violations += rule(it)

    lines = numbered_lines(report.answer)
    report.checklist = await build_checklist(it, rq, usage)
    if report.checklist:
        report.grading = await grade(it, report.checklist, rq, lines, usage)
    if report.checklist and report.grading:
        report.coverage = coverage_score(report.checklist, report.grading)
        report.focus = focus_score(lines, report.grading)

    if should_ground(it, report.grading, report.rule_violations):
        report.grounding = await ground(it, lines, usage)
        if report.grounding:
            report.correctness = correctness_score(report.grounding, report.rule_violations)

    report.findings = extract_findings(report, lines, it)
    report.escalation_reasons = escalation(report)
    report.escalate = bool(report.escalation_reasons)
    report.reviewer_tokens = usage
    return report

# ============================== Dispatch ==============================
coordinator_inbox: list[EvaluationReport] = []


async def evaluate_and_dispatch(it: Interaction, rq: str, run_id: Optional[str],
                                run_dir: Optional[Path]) -> None:
    try:
        report = await evaluate(it, rq, run_id)
    except Exception:
        logger.exception("Review failed for %s", it.interaction_id)
        return

    if config.WRITE_REPORTS and run_dir:
        d = Path(run_dir) / "reports"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{report.agent_name}_{report.report_id}.json").write_text(json.dumps(
            {"report": report.model_dump(), "trace": it.trace_text()}, indent=1, default=str))
    if config.FEED_COORDINATOR:
        coordinator_inbox.append(report)
    if config.feeds_hypotheses():
        ledger.add_findings(report.findings, run_id)
    if config.ESCALATE and report.escalate:
        notify_human(report)


# Keep references to background tasks or they can be garbage-collected mid-run.
_background: set[asyncio.Task] = set()


def spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def pending_count() -> int:
    return len(_background)


async def drain(timeout: Optional[float] = None) -> None:
    """Await pending reviews."""
    if _background:
        await asyncio.wait(set(_background), timeout=timeout)


def get_evaluation_feedback() -> dict:
    """Returns review feedback on recent sub-agent answers: missing essential content,
    claims contradicted by tool outputs, procedural issues, and problems with how the
    question was phrased. Call before deciding whether to re-ask a sub-agent or move on."""
    items = []
    while coordinator_inbox:
        r = coordinator_inbox.pop(0)
        missing = []
        if r.checklist and r.grading:
            by_id = {v.item_id: v.present for v in r.grading.verdicts}
            missing = [i.description for i in r.checklist.items
                       if i.importance == "essential" and by_id.get(i.id) != "yes"]
        items.append({  # DECISION: what the coordinator sees; keep it actionable
            "agent": r.agent_name,
            "question": r.question[:300],
            "scores": {"coverage": r.coverage, "focus": r.focus, "correctness": r.correctness},
            "missing_essential": missing,
            "contradicted_claims": [c.claim for c in (r.grounding.claims if r.grounding else [])
                                    if c.status == "contradicted"],
            "procedural_issues": r.rule_violations,
            "question_issues": r.checklist.question_issues if r.checklist else [],
        })
    return {"reports": items, "evaluations_still_running": pending_count()}