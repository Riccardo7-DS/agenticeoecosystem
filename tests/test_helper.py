# Run with: AGENTICEO_SUBAGENTS=none PYTHONPATH=src uv run --with pytest python -m pytest tests/test_helper.py -q
from types import SimpleNamespace

import pytest

from agenticeoecosystem.agents import helper
from agenticeoecosystem.agents.helper import ledger as L
from agenticeoecosystem.agents.helper import reviewer as ev
from agenticeoecosystem.agents.helper import rules
from agenticeoecosystem.agents.helper.schemas import (
    Checklist, ChecklistItem, ClaimCheck, EvaluationReport, Finding, Grading,
    Grounding, ItemVerdict, LineLabel,
)
from agenticeoecosystem.agents.helper.trace import Interaction, TraceStep
from agenticeoecosystem.agents.orchestrator_common import (
    SUPERVISOR_OUTPUT_RULES, SUPERVISOR_ROUTING, SUPERVISOR_SYSTEM_PROMPT,
)

# ---------------- wiring ----------------

def test_prompt_split_is_lossless():
    assert SUPERVISOR_ROUTING + SUPERVISOR_OUTPUT_RULES == SUPERVISOR_SYSTEM_PROMPT
    assert SUPERVISOR_SYSTEM_PROMPT.count("OUTPUT RULES:") == 1


def test_off_mode_passes_only_on_tool_error():
    helper.configure("off")
    f = lambda **kw: None
    assert helper.root_callbacks(f) == {"on_tool_error_callback": f}


def test_review_mode_is_valid_and_unknown_modes_fail():
    try:
        helper.configure("review")
        assert helper.config.MODE == "review"
        with pytest.raises(ValueError):
            helper.configure("reveiw")
    finally:
        helper.configure("off")


@pytest.fixture
def review_hooks(monkeypatch):
    """hooks in review mode, with reviews captured instead of run."""
    from agenticeoecosystem.agents.helper import hooks
    helper.configure("review")
    spawned = []
    monkeypatch.setattr(hooks, "evaluate_and_dispatch", lambda it, *a: it)
    monkeypatch.setattr(hooks, "spawn", spawned.append)
    monkeypatch.setattr(hooks, "run", hooks.RunState())
    yield hooks, spawned
    helper.configure("off")


def _call_subagent(hooks, name, response):
    """What ADK does for a mode='single_turn' sub-agent: a root tool named after the agent."""
    tool = SimpleNamespace(name=name)
    hooks.root_before_tool(tool, {"request": "Mean t2m over Paris in July 2023?"}, None)
    hooks.sub_before_agent(SimpleNamespace(agent_name=name, user_content=None))
    hooks.root_after_tool(tool, {"request": "..."}, None, response)


def test_subagent_call_is_recognised_by_agent_name(review_hooks):
    hooks, spawned = review_hooks
    _call_subagent(hooks, "climate_agent", "Mean t2m was 293 K.")
    assert len(spawned) == 1
    it = spawned[0]
    assert it.question == "Mean t2m over Paris in July 2023?"
    assert it.question_source == "task_tool_args"
    assert it.answer == "Mean t2m was 293 K."


def test_other_root_tools_are_ignored(review_hooks):
    hooks, spawned = review_hooks
    hooks.root_before_tool(SimpleNamespace(name="get_hypothesis_summary"), {}, None)
    assert hooks.run.pending_questions == {}


def test_failed_subagent_call_is_not_reviewed(review_hooks):
    hooks, spawned = review_hooks
    _call_subagent(hooks, "climate_agent", "Error running sub-agent: CDS request failed")
    assert spawned == []
    assert not hooks.run.open["climate_agent"]   # discarded, not left open


def test_subagent_with_no_answer_is_not_reviewed(review_hooks):
    """What the root gets when the sub-agent's own model call fails (seen with ADK 2.2)."""
    hooks, spawned = review_hooks
    _call_subagent(hooks, "climate_agent", {"result": None})
    assert spawned == []
    assert not hooks.run.open["climate_agent"]


def test_chain_keeps_existing_result():
    calls = []
    chained = helper._chain(lambda **kw: {"error": "x"}, lambda **kw: calls.append("ours"))
    assert chained(tool=None) == {"error": "x"}
    assert calls == ["ours"]

# ---------------- scoring and rules ----------------

def test_missing_essential_caps_coverage():
    cl = Checklist(items=[ChecklistItem(id="a", description="x", importance="essential"),
                          ChecklistItem(id="b", description="y", importance="useful")])
    g = Grading(verdicts=[ItemVerdict(item_id="a", present="no"), ItemVerdict(item_id="b", present="yes")])
    assert ev.coverage_score(cl, g) <= 2


def test_unexpected_is_not_noise():
    g = Grading(verdicts=[], off_topic_lines=[LineLabel(line=1, relevance="unexpected_but_valuable")])
    assert ev.focus_score(["on topic", "surprising but useful"], g) == 5


def test_precip_mean_rule():
    it = Interaction(agent_name="climate_agent", answer="", steps=[
        TraceStep(0, "convert_hourly_to_daily", {"variable": "total_precipitation", "method": "mean"}, "")])
    assert rules._climate_precip_sum_rule(it)


def test_fabricated_path_rule():
    it = Interaction(agent_name="climate_agent", answer="Saved to data/fake.nc", steps=[
        TraceStep(0, "fetch_era5_single_levels", {"output_filename": "data/real.nc"}, "ok")])
    assert rules._files_exist_rule(it)


def test_arxiv_abstract_cap():
    it = Interaction(agent_name="arxiv_rag_agent", answer="",
                     steps=[TraceStep(i, "get_abstract", {}, "") for i in range(6)])
    assert rules._arxiv_abstract_cap(it)


def test_arxiv_zero_retry():
    it = Interaction(agent_name="arxiv_rag_agent", answer="",
                     steps=[TraceStep(0, "search_papers", {}, '{"total_results": 0, "papers": []}')])
    assert rules._arxiv_zero_retry(it)


def test_findings_reliability_and_leads():
    lines = ["mean t2m was 285 K", "SST anomaly unusually high in March"]
    it = Interaction(agent_name="climate_agent", steps=[TraceStep(0, "summarize_regional_timeseries", {}, "")])
    r = EvaluationReport(
        report_id="r1", run_id="run", agent_name="climate_agent", question="q",
        question_source="task_tool_args", answer="", created_at=0, rule_violations=["step 0: x"],
        grading=Grading(verdicts=[], off_topic_lines=[LineLabel(line=1, relevance="unexpected_but_valuable")]),
        grounding=Grounding(claims=[ClaimCheck(claim=lines[0], status="supported",
                                               claim_type="result", lines=[0], trace_step=0)]))
    by_kind = {f.kind: f for f in ev.extract_findings(r, lines, it)}
    assert by_kind["result"].reliability == "medium"   # downgraded by the rule violation
    assert by_kind["lead"].reliability == "unverified"

# ---------------- ledger ----------------

@pytest.fixture
def led(tmp_path):
    L.ledger.reset("run", tmp_path)
    return L.ledger


def _f(fid, rid, kind="result", rel="high"):
    return Finding(finding_id=fid, report_id=rid, run_id="run", agent_name="a",
                   question="q", text=fid, kind=kind, reliability=rel)


def _h():
    return L.propose_hypothesis("H", "general", ["p"], "f", "r", [], "")["created"]


def test_same_report_counts_once(led):
    led.add_findings([_f("f1", "r1"), _f("f2", "r1")], "run")
    h = _h()
    L.record_evidence(h, "f1", "supports", "strong", "x")
    L.record_evidence(h, "f2", "supports", "strong", "x")
    assert L.score(led.hypotheses[h])["pro"] == 2.0


def test_supported_needs_two_reports(led):
    led.add_findings([_f("f1", "r1")], "run")
    h = _h()
    L.record_evidence(h, "f1", "supports", "decisive", "x")
    assert "error" in L.set_status(h, "supported", "x")


def test_lead_is_not_evidence(led):
    led.add_findings([_f("f1", "r1", kind="lead", rel="unverified")], "run")
    assert "error" in L.record_evidence(_h(), "f1", "supports", "weak", "x")


def test_refine_downweights_old_evidence(led):
    led.add_findings([_f("f1", "r1")], "run")
    h = _h()
    L.record_evidence(h, "f1", "supports", "strong", "x")
    L.revise_hypothesis(h, "refine", "", "summer only", [], "", "seasonal", [])
    assert L.score(led.hypotheses[h])["pro"] == 1.0


def test_late_findings_from_old_run_dropped(led):
    led.add_findings([_f("f1", "r1")], "older-run")
    assert "f1" not in led.findings