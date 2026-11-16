"""End-to-end Virtual Lab meetings against a scripted LLM and fake sub-agents.

Covers agent collaboration (independent proposals, peer/critic review,
revision), evidence tracking (tool-call provenance, downgraded claims),
plan_only permissions, and termination (convergence, max rounds, budget).
The specialists, critic, PI and designer are real ADK LlmAgents; existing
sub-agents are reached through real AgentTools.

Run:
    uv run --with pytest pytest tests/test_virtual_lab_meeting.py
"""

import asyncio
import json

import pytest

from _virtual_lab_fakes import (
    OPTICAL_V0,
    PROBLEM,
    QUERY,
    THERMAL_V1,
    ScriptedLlm,
    _call,
    _has_tool_result,
    _request_text,
    _text,
    default_script,
    fake_existing_agents,
)
from agenticeoecosystem.agents.virtual_lab import discovery
from agenticeoecosystem.agents.virtual_lab.config import VirtualLabConfig
from agenticeoecosystem.agents.virtual_lab.inventory import Catalogs
from agenticeoecosystem.agents.virtual_lab.literature import rank_from_payload
from agenticeoecosystem.agents.virtual_lab.runner import run_virtual_lab
from agenticeoecosystem.agents.virtual_lab.schemas import DecisionStatus, EvidenceKind


def _run(tmp_path, script=default_script, **cfg):
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(**{"participants": ["thermal_specialist", "optical_specialist"],
                                 "literature_enabled": False, **cfg})
    answer, record, ledger, agents = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path))
    return answer, record, ledger, agents, llm


def _requests(llm, role, phase):
    return [text for r, p, text in llm.seen if r == role and p == phase]


def test_full_meeting_produces_validated_decision_and_plan(tmp_path):
    answer, record, ledger, agents, _ = _run(tmp_path)

    assert record.problem.study_area.bbox == [-72.73, -37.41, -72.59, -37.22]
    assert set(record.final_proposals) == {"thermal_specialist", "optical_specialist"}
    assert record.termination == "max_rounds" and record.rounds_run == 2
    assert record.decision.status == DecisionStatus.DECIDED
    assert record.plan is not None and record.plan.workflow[0].agent == "gee_agent"
    assert answer.startswith("# Virtual Lab decision: decided")
    assert {"lab_pi", "thermal_specialist", "optical_specialist", "scientific_critic",
            "experiment_designer"} <= set(agents)
    # The record is persisted as an external artifact, not stuffed into state.
    saved = list(tmp_path.rglob("virtual_lab_record.json*"))
    assert saved, "record artifact not written"


def test_proposals_are_independent_then_shared(tmp_path):
    *_, llm = _run(tmp_path)
    thermal_propose = "\n".join(_requests(llm, "thermal_specialist", "propose"))
    optical_propose = "\n".join(_requests(llm, "optical_specialist", "propose"))
    assert "OPTICAL-V0" not in thermal_propose
    assert "THERMAL-V0" not in optical_propose
    # Phase C shares proposals: each reviewer sees the other's, the reviser sees critiques.
    assert "OPTICAL-V0" in "\n".join(_requests(llm, "thermal_specialist", "critique"))
    revise = "\n".join(_requests(llm, "thermal_specialist", "revise"))
    assert "Chile is at the edge of the 0 deg disk" in revise   # critic finding
    assert "viewing zenith" in revise                           # validator finding


def test_critic_and_validator_drive_revision(tmp_path):
    _, record, *_ = _run(tmp_path)
    r1 = [c for c in record.critiques if c.round == 1 and c.target == "thermal_specialist"]
    authors = {c.author for c in r1}
    assert {"scientific_critic", "validator"} <= authors
    assert any(i.category == "coverage" and i.severity == "blocking" for c in r1 for i in c.issues)
    thermal = record.final_proposals["thermal_specialist"]
    assert thermal.summary == THERMAL_V1["summary"] and thermal.revision >= 1
    assert all(d.name != "MTG FCI" for d in thermal.datasets)
    # Hypothesis ids are namespaced per specialist so the PI can select them unambiguously.
    assert thermal.hypotheses[0].id == "thermal-H1"


def test_evidence_is_linked_to_tool_calls(tmp_path):
    _, record, ledger, *_ = _run(tmp_path)
    thermal = record.final_proposals["thermal_specialist"]
    ev = thermal.evidence[0]
    assert ev.kind == EvidenceKind.RETRIEVED and ev.tool_call_ids
    calls = {c["id"]: c for c in ledger.tool_calls}
    [cited] = [calls[i] for i in ev.tool_call_ids]
    assert cited["tool"] == "eumetsat_agent" and cited["step"] == "r2.revise.thermal_specialist"
    # The revision repeated the proposal's request, so the cache answered it; the original
    # call and the leaf MCP call it made remain the provenance.
    original = calls[cited["cached_from"]]
    assert original["step"] == "r0.propose.thermal_specialist" and original["status"] == "ok"
    leaves = [c for c in ledger.tool_calls if c["via"] == "eumetsat_agent" and c["step"] == original["step"]]
    assert [c["tool"] for c in leaves] == ["search_eumetsat_products"]
    # The optical claim had no backing call: downgraded, never silently trusted.
    optical = record.final_proposals["optical_specialist"]
    assert optical.evidence[0].kind == EvidenceKind.ASSUMPTION
    links = {(p.relation, p.source) for p in record.decision.provenance}
    assert ("based_on_proposal", "proposal:thermal_specialist:rev2") in links
    assert any(rel == "cites_tool_call" for rel, _ in links)


def test_plan_only_blocks_downloading_tools(tmp_path):
    def script(role, phase, req):
        if role == "optical_specialist" and phase == "propose" and not _has_tool_result(req):
            return _call("cdse_agent", {"request": "download S2 over the AOI"})
        return default_script(role, phase, req)

    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    blocked = [c for c in ledger.tool_calls if c["status"] == "blocked"]
    assert [c["tool"] for c in blocked] == ["fetch_sentinel2"]
    assert blocked[0]["step"] == "r0.propose.optical_specialist"
    assert "plan_only" in blocked[0]["reason"]


def test_converges_early_when_no_open_issues(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "propose" and _has_tool_result(req):
            return _text(THERMAL_V1)  # already clean: VIIRS + SLSTR, grounded
        if role == "scientific_critic":
            return _text({"critiques": []})
        return default_script(role, phase, req)

    _, record, *_ = _run(tmp_path, script=script, participants=["thermal_specialist"], max_rounds=3)
    assert record.termination == "converged" and record.rounds_run == 1
    assert not [c for c in record.proposal_history if c.revision > 0]


def test_budget_bounds_the_meeting(tmp_path):
    _, record, ledger, *_ = _run(tmp_path, max_llm_calls=5)
    assert record.termination == "budget"
    assert ledger.lab_llm_calls() <= 5
    assert record.decision.status != DecisionStatus.DECIDED


def test_tool_call_budget_is_enforced(tmp_path):
    _, record, ledger, *_ = _run(tmp_path, max_tool_calls=1, max_rounds=0)
    assert ledger.budget_exhausted == "tool_calls"
    assert any(c["status"] == "blocked" and "budget" in c["reason"] for c in ledger.tool_calls)


def test_invalid_output_is_repaired_once(tmp_path):
    attempts = {"n": 0}

    def script(role, phase, req):
        if role == "experiment_designer":
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _text("Here is my plan: step one, download everything.")
            assert "REJECTED" in json.dumps(str(req.config.system_instruction))
        return default_script(role, phase, req)

    _, record, *_ = _run(tmp_path, script=script, max_rounds=0)
    assert record.plan is not None
    assert any(o["agent"] == "experiment_designer" for o in record.invalid_outputs)


def test_sequential_mode_keeps_proposals_independent(tmp_path):
    *_, llm = _run(tmp_path, parallel=False, max_rounds=0)
    assert "THERMAL-V0" not in "\n".join(_requests(llm, "optical_specialist", "propose"))


PAPERS = [{"id": f"2101.0000{n}", "title": f"Paper {n}", "abstract": f"Smoke biases dNBR, case {n}.",
           "published": "2021-01-01"} for n in (1, 2, 3, 4)]
OFF_TOPIC = {"id": "cond-mat/0411570", "title": "Spin glasses", "abstract": "Smoke rings in superfluids.",
             "published": "2004-11-22"}
CITATIONS = [{"citationCount": 5, "year": 2021}, {"citationCount": 50, "year": 2021}, None,
             {"citationCount": 20, "year": 2021}]


def test_contested_claim_triggers_citation_ranked_literature_check(tmp_path):
    searches, ranked_ids = [], []

    async def search(ctx, query, n):
        searches.append(query)
        return [dict(p) for p in PAPERS + [OFF_TOPIC]]

    async def rank(ids):
        ranked_ids.append(ids)
        return rank_from_payload(ids, CITATIONS)

    def script(role, phase, req):
        if role == "literature_reviewer" and phase == "lit_query":
            return _text({"claims": [{"claim_id": "r1-c1", "scope": "general",
                                      "queries": ['"dNBR" AND smoke']}]})
        if role == "literature_reviewer" and phase == "lit_assess":
            return _text({"assessments": [{"claim_id": "r1-c1", "verdict": "supports",
                                           "summary": "SMOKE-DEGRADES-DNBR", "papers": ["2101.00002"]}]})
        if role == "optical_specialist" and phase == "revise":
            return _text({**OPTICAL_V0, "summary": "OPTICAL-V1", "evidence": [
                {"claim": "smoke degrades dNBR", "kind": "retrieved", "source": "arxiv:2101.00002"}]})
        return default_script(role, phase, req)

    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        literature_search=search, citation_ranker=rank))

    # Round 1: thermal disputes optical's "S2 dNBR" -> one check; round 2 repeats it -> not re-checked.
    assert len(record.literature) == 1 and searches == ['"dNBR" AND smoke']
    check = record.literature[0]
    assert (check.raised_by, check.target, check.verdict) == ("thermal_specialist", "optical_specialist", "supports")
    assert [p.arxiv_id for p in check.papers] == ["2101.00002", "2101.00004", "2101.00001"]  # by citations
    assert "cond-mat/0411570" not in ranked_ids[0]  # mentions only one query term: dropped before ranking
    assert [p.citation_count for p in check.papers] == [50, 20, 5]
    tools = {c["id"]: c["tool"] for c in ledger.tool_calls}
    assert sorted(tools[i] for i in check.tool_call_ids) == ["rank_papers_by_citations", "search_papers"]
    # Specialists revise with the findings and can cite the papers as linked, retrieved evidence.
    assert "SMOKE-DEGRADES-DNBR" in "\n".join(_requests(llm, "optical_specialist", "revise"))
    evidence = record.final_proposals["optical_specialist"].evidence[0]
    assert evidence.kind == EvidenceKind.RETRIEVED
    assert [tools[i] for i in evidence.tool_call_ids] == ["search_papers"]


@pytest.mark.parametrize("limits", [{"max_literature_tool_calls": 1},
                                    {"max_literature_tool_calls": 12, "max_literature_tool_calls_per_check": 1}])
def test_literature_check_has_its_own_tool_allowance(tmp_path, limits):
    searches, ranked_ids = [], []

    async def search(ctx, query, n):
        searches.append(query)
        return [dict(p) for p in PAPERS]

    async def rank(ids):
        ranked_ids.append(ids)
        return rank_from_payload(ids, CITATIONS)

    def script(role, phase, req):
        if role == "literature_reviewer" and phase == "lit_query":
            return _text({"claims": [{"claim_id": "r1-c1", "scope": "general",
                                      "queries": ['"dNBR" AND smoke']}]})
        return default_script(role, phase, req)

    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_tool_calls=0, **limits)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        literature_search=search, citation_ranker=rank))

    # The agents' budget is 0, yet the check still searches; its own allowance (1) stops the ranking.
    assert searches == ['"dNBR" AND smoke'] and ranked_ids == []
    assert record.literature[0].papers                         # kept in search order
    assert ledger.literature_tool_calls() == 1 and ledger.used_tool_calls() == 0
    assert ledger.usage()["literature_tool_calls"] == 1


def test_unresponsive_arxiv_cannot_stall_the_meeting(tmp_path):
    searches = []

    async def hanging_search(ctx, query, n):
        searches.append(query)
        await asyncio.sleep(30)            # arXiv not answering
        return []

    def script(role, phase, req):
        if role == "literature_reviewer" and phase == "lit_query":
            return _text({"claims": [{"claim_id": "r1-c1", "scope": "general",
                                      "queries": ['"dNBR" AND smoke', "smoke dNBR bias"]}]})
        return default_script(role, phase, req)

    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=1, literature_timeout_s=0.2)
    _, record, ledger, _ = asyncio.run(asyncio.wait_for(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        literature_search=hanging_search, citation_ranker=None), timeout=60))

    assert searches == ['"dNBR" AND smoke']                     # second query skipped: arXiv is down
    [call] = [c for c in ledger.tool_calls if c["tool"] == "search_papers"]
    assert call["status"] == "error" and call["timed_out"]
    check = record.literature[0]
    assert check.verdict == "inconclusive" and "arXiv did not respond" in check.summary
    assert record.decision is not None and record.plan is not None


def test_critic_plan_review_triggers_redesign(tmp_path):
    def script(role, phase, req):
        if role == "scientific_critic" and phase == "plan_review":
            return _text({"critiques": [{"target": "experiment_plan", "issues": [{
                "category": "methodology", "severity": "major", "claim": "thermal-H1",
                "rationale": "PREDICTION-CONTRADICTS-STATEMENT"}]}]})
        return default_script(role, phase, req)

    _, record, *_, llm = _run(tmp_path, script=script)
    design_calls = _requests(llm, "experiment_designer", "design")
    assert len(design_calls) == 2 and "PREDICTION-CONTRADICTS-STATEMENT" in design_calls[1]
    assert any(c.author == "scientific_critic" and c.target == "experiment_plan" for c in record.critiques)


def test_empty_search_is_broadened_once_and_identical_queries_are_cached(tmp_path):
    searches = []

    async def search(ctx, query, n):
        searches.append(query)
        return [] if '"' in query else [dict(p) for p in PAPERS]

    async def rank(ids):
        return rank_from_payload(ids, CITATIONS)

    def script(role, phase, req):
        if role == "literature_reviewer" and phase == "lit_query":
            return _text({"claims": [{"claim_id": "r1-c1", "scope": "general",
                                      "queries": ['"dNBR bias" AND smoke', '"dNBR bias" AND smoke']}]})
        if role == "literature_reviewer" and phase == "lit_assess":
            return _text({"assessments": [{"claim_id": "r1-c1", "verdict": "mixed", "summary": "s",
                                           "papers": ["2101.00001"]}]})
        return default_script(role, phase, req)

    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        literature_search=search, citation_ranker=rank))

    assert searches == ['"dNBR bias" AND smoke', "dNBR bias OR smoke"]  # retry once, then cached
    check = record.literature[0]
    assert check.queries[-1] == "dNBR bias OR smoke" and len(check.papers) == 3
    assert len(set(check.tool_call_ids)) == len(check.tool_call_ids) == 3  # 2 searches + 1 ranking


def test_revision_keeps_linked_evidence_it_did_not_restate(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "revise":  # no tool call, evidence rewritten away
            return _text({**THERMAL_V1, "evidence": []})
        return default_script(role, phase, req)

    _, record, ledger, *_ = _run(tmp_path, script=script)
    evidence = record.final_proposals["thermal_specialist"].evidence
    linked = [e for e in evidence if e.kind == EvidenceKind.RETRIEVED and e.tool_call_ids]
    assert len(linked) == 1  # carried from r0, not duplicated across the two revisions
    steps = {c["id"]: c["step"] for c in ledger.tool_calls}
    assert {steps[i] for i in linked[0].tool_call_ids} == {"r0.propose.thermal_specialist"}


def test_designer_is_told_lab_members_do_not_execute(tmp_path):
    *_, llm = _run(tmp_path, max_rounds=0)
    design = _requests(llm, "experiment_designer", "design")[0]
    assert "never a lab member" in design and "thermal_specialist" in design.split("execute nothing")[0]


def test_spent_tool_budget_still_lets_the_meeting_finish(tmp_path):
    _, record, ledger, *_ = _run(tmp_path, max_tool_calls=1)
    assert ledger.budget_exhausted == "tool_calls"
    assert record.rounds_run == 2 and record.termination == "max_rounds"  # revisions still ran
    assert record.plan is not None


def test_tool_call_is_kept_as_evidence_when_the_model_labels_nothing_retrieved(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "propose" and _has_tool_result(req):
            return _text({**THERMAL_V1, "evidence": [{"claim": "FCI sees the fire", "kind": "assumption"}]})
        return default_script(role, phase, req)

    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    [proposal] = [p for p in record.proposal_history if p.agent == "thermal_specialist"]
    auto = [e for e in proposal.evidence if e.kind == EvidenceKind.RETRIEVED]
    assert len(auto) == 1 and auto[0].source == "eumetsat_agent"
    assert auto[0].claim.startswith("eumetsat_agent returned:")
    tools = {c["id"]: c["tool"] for c in ledger.tool_calls}
    assert {tools[i] for i in auto[0].tool_call_ids} == {"eumetsat_agent", "search_eumetsat_products"}


def test_revision_links_a_restated_claim_to_the_earlier_call(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "revise":  # restates the r0 finding, no new call
            return _text(THERMAL_V1)
        return default_script(role, phase, req)

    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=1)
    [ev] = record.final_proposals["thermal_specialist"].evidence
    assert ev.kind == EvidenceKind.RETRIEVED  # not downgraded to an assumption
    steps = {c["id"]: c["step"] for c in ledger.tool_calls}
    assert {steps[i] for i in ev.tool_call_ids} == {"r0.propose.thermal_specialist"}


def _function_responses(req) -> int:
    return sum(1 for c in req.contents for p in (c.parts or []) if p.function_response)


def test_sub_agent_reply_without_data_is_not_evidence(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "propose" and not _has_tool_result(req):
            return _call("gee_agent", {"request": "VIIRS active fire over Chile 2026-01-17"})
        if role == "gee_agent":
            return _text("I'm sorry, but I don't have access to data for future dates.")
        if role == "thermal_specialist" and phase == "propose":
            seen_responses.extend(str(p.function_response.response) for c in req.contents
                                  for p in (c.parts or []) if p.function_response)
        return default_script(role, phase, req)

    seen_responses: list[str] = []
    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    [call] = [c for c in ledger.tool_calls if c["tool"] == "gee_agent"]
    assert call["status"] == "no_answer"
    assert call["args"]["request"].startswith("[Today is ")
    [proposal] = [p for p in record.proposal_history if p.agent == "thermal_specialist"]
    assert not [e for e in proposal.evidence if call["id"] in e.tool_call_ids]
    assert any("Lab note" in r for r in seen_responses)


def test_prompts_carry_todays_date(tmp_path):
    from agenticeoecosystem.agents.virtual_lab.prompts import today
    *_, llm = _run(tmp_path, max_rounds=0)
    assert all(f"TODAY: {today()}" in t for t in _requests(llm, "thermal_specialist", "propose"))


def test_critique_sub_agent_calls_are_capped(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "critique" and (n := _function_responses(req)) < 4:
            return _call("eumetsat_agent", {"request": f"check FCI again, query {n}"})  # no cache hits
        return default_script(role, phase, req)

    _, _, ledger, *_ = _run(tmp_path, script=script, max_rounds=1, max_critique_tool_calls=2)
    step = [c for c in ledger.tool_calls if c["step"] == "r1.critique.thermal_specialist" and c["via"] is None]
    assert sum(c["status"] == "ok" for c in step) == 2
    assert any(c["status"] == "blocked" and "Critique tool limit" in c["reason"] for c in step)


def test_pi_formulation_is_limited_to_one_tool_call(tmp_path):
    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            assert "Do NOT ask about data availability" in _request_text(req)
            if (n := _function_responses(req)) < 3:
                return _call("cdse_agent", {"request": f"is Sentinel-2 available? query {n}"})
        return default_script(role, phase, req)

    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    step = [c for c in ledger.tool_calls if c["step"] == "r0.formulate" and c["via"] is None]
    assert sum(c["status"] != "blocked" for c in step) == 1
    assert all("Formulation tool limit (1 calls)" in c["reason"]
               for c in step if c["status"] == "blocked")
    assert record.problem is not None


def test_proposal_sub_agent_calls_are_capped(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "propose" and (n := _function_responses(req)) < 4:
            return _call("eumetsat_agent", {"request": f"check FCI again, query {n}"})  # no cache hits
        return default_script(role, phase, req)

    _, _, ledger, *_ = _run(tmp_path, script=script, max_rounds=0, max_proposal_tool_calls=2)
    step = [c for c in ledger.tool_calls if c["step"] == "r0.propose.thermal_specialist" and c["via"] is None]
    assert sum(c["status"] == "ok" for c in step) == 2
    assert any(c["status"] == "blocked" and "Proposal tool limit" in c["reason"] for c in step)


def test_blocked_calls_do_not_use_the_tool_budget():
    from agenticeoecosystem.agents.virtual_lab.ledger import LabLedger
    ledger = LabLedger()
    for status in ("ok", "blocked", "blocked", "no_answer"):
        ledger.add_tool_call(step="s", tool="t", status=status)
    assert ledger.used_tool_calls() == 2


def test_future_date_reply_is_not_evidence_even_after_tool_calls(tmp_path):
    def script(role, phase, req):
        if role == "eumetsat_agent" and _has_tool_result(req):
            return _text("Found 144 products, but the date range is in the future so they are placeholders.")
        return default_script(role, phase, req)

    _, _, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    [call] = [c for c in ledger.tool_calls if c["tool"] == "eumetsat_agent"]
    assert call["status"] == "no_answer"


class _FakeCatalog:
    """Offline stand-in for servers.cdse_geocode.catalog."""
    calls: list = []

    @classmethod
    async def search_scenes(cls, collection, bbox, start_date, end_date, product_type=None, limit=20, **_):
        cls.calls.append(collection)
        if collection == "SENTINEL-3":
            return {"error": "CDSE catalogue search failed: timeout"}
        return {"collection": collection, "product_type": product_type, "bbox": bbox,
                "start_date": start_date, "end_date": end_date, "total": 1, "listed": 1,
                "products": [{"name": f"{collection}_X", "start": "2026-01-17T14:37:29Z"}]}

    @classmethod
    async def list_stac_collections(cls, bbox=None, start_date=None, end_date=None):
        cls.calls.append("stac")
        return {"total": 2, "matching": 2, "collections": [
            {"id": "clms_burntarea_global_300m_daily_v4", "title": "CLMS Burnt Area Global 300m daily V4",
             "description": "Burned area detected from satellite observations", "keywords": ["fire disturbance"]},
            {"id": "clms_swe_northernhemisphere_5km_daily_v2", "title": "CLMS Snow Water Equivalent",
             "description": "Snow water equivalent", "keywords": ["cryosphere"]}]}

    @classmethod
    async def search_stac_items(cls, collection, bbox, start_date, end_date, **_):
        cls.calls.append("stac_items")
        return {"collection": collection, "start_date": start_date, "end_date": end_date, "total": None,
                "listed": 2, "days": {"2026-01-17": ["14:00"], "2026-01-18": ["14:00"]}}

    @classmethod
    async def list_openeo_collections(cls, bbox=None, start_date=None, end_date=None):
        cls.calls.append("openeo")
        return {"total": 3, "matching": 2, "collections": [
            {"id": "CLMS_BA_GLOBAL_300M_DAILY_V4", "title": "CLMS Burnt Area (BA) Global 300m daily V4",
             "keywords": ["fire disturbance"]},
            {"id": "CLMS_SWE_NORTHERNHEMISPHERE_5KM_DAILY_V2", "title": "CLMS Snow Water Equivalent",
             "keywords": ["cryosphere"]}]}


def test_data_inventory_is_built_once_and_shown_to_everyone(tmp_path):
    _FakeCatalog.calls = []
    llm = ScriptedLlm(script=default_script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=1)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(cdse=_FakeCatalog)))

    inventory = [c for c in ledger.tool_calls if c.get("inventory")]
    # 6 scene searches + STAC listing + item search of the one relevant collection
    assert len(inventory) == 8 and sorted(_FakeCatalog.calls).count("stac_items") == 1
    assert len(_FakeCatalog.calls) == 10                 # the two failing searches are retried once
    assert {c["step"] for c in inventory} == {"r0.inventory"}
    s2 = next(c for c in inventory if c["args"].get("collection") == "SENTINEL-2")
    assert s2["status"] == "ok" and s2["catalog"]["dates"] == ["2026-01-17"]
    assert s2["args"]["start_date"] == "2026-01-03"                        # padded window
    assert sum(c["status"] == "error" for c in inventory) == 2            # both Sentinel-3 searches
    for role, phase in [("thermal_specialist", "propose"), ("scientific_critic", "critique"),
                        ("lab_pi", "decide"), ("experiment_designer", "design")]:
        text = "\n".join(_requests(llm, role, phase))
        assert "DATA INVENTORY" in text and "clms_burntarea_global_300m_daily_v4" in text, (role, phase)
    propose = "\n".join(_requests(llm, "thermal_specialist", "propose"))
    assert "NOT absent" in propose and "clms_swe" not in propose       # unrelated collection left out
    assert "DATA DISCOVERY" in propose and "word overlap" in propose     # tests rank without embeddings
    assert "DATA INVENTORY (" not in "\n".join(_requests(llm, "lab_pi", "formulate"))


def test_no_inventory_without_a_catalog(tmp_path):
    *_, ledger, _, llm = _run(tmp_path, max_rounds=0)
    assert not any(c.get("inventory") for c in ledger.tool_calls)
    assert "DATA INVENTORY (" not in "\n".join(_requests(llm, "thermal_specialist", "propose"))


@pytest.fixture(autouse=True)
def _fresh_discovery_cache():
    """discovery caches catalogue listings per process: each test lists its own fakes."""
    discovery._LIST_CACHE.clear()
    yield
    discovery._LIST_CACHE.clear()


def _tool_names(req) -> set[str]:
    return set(getattr(req, "tools_dict", {}) or {})


def test_repeated_sub_agent_request_is_answered_from_cache(tmp_path):
    # The default thermal specialist asks eumetsat_agent the same question when revising.
    _, record, ledger, *_ = _run(tmp_path, max_rounds=1)
    cached = [c for c in ledger.tool_calls if c.get("cached_from")]
    assert [c["tool"] for c in cached] == ["eumetsat_agent"]
    assert cached[0]["step"] == "r1.revise.thermal_specialist" and cached[0]["status"] == "ok"
    leaf = [c for c in ledger.tool_calls if c["tool"] == "search_eumetsat_products"]
    assert len(leaf) == 1                                   # the sub-agent ran once
    assert ledger.used_tool_calls() == sum(1 for c in ledger.tool_calls
                                           if c["status"] != "blocked") - 1
    assert ledger.usage()["cached_tool_calls"] == 1


def test_sub_agent_reply_without_a_successful_tool_call_is_not_evidence(tmp_path):
    def script(role, phase, req):
        if role == "optical_specialist" and phase == "propose" and not _has_tool_result(req):
            return _call("cdse_agent", {"request": "Sentinel-2 L2A over the bbox on 2026-01-17"})
        return default_script(role, phase, req)   # cdse_agent's only tool is blocked (fetch)
    _, record, ledger, *_ = _run(tmp_path, script=script, max_rounds=0)
    [cdse] = [c for c in ledger.tool_calls if c["tool"] == "cdse_agent"]
    assert cdse["status"] == "no_answer"
    [fetch] = [c for c in ledger.tool_calls if c["tool"] == "fetch_sentinel2"]
    assert fetch["status"] == "blocked" and fetch["via"] == "cdse_agent"   # blocked leaf keeps via
    optical = record.final_proposals["optical_specialist"]
    assert not any(cdse["id"] in e.tool_call_ids for e in optical.evidence)


def test_sub_agent_run_tool_calls_are_capped(tmp_path):
    def script(role, phase, req):
        if role == "eumetsat_agent":
            done = sum(1 for c in req.contents for p in (c.parts or []) if p.function_response)
            if done < 5:
                return _call("search_eumetsat_products", {"collection": "FCI", "date": f"2026-01-1{done}"})
            return _text("done")
        return default_script(role, phase, req)
    _, _, ledger, *_ = _run(tmp_path, script=script, max_rounds=0, max_sub_agent_tool_calls=3)
    leaf = [c for c in ledger.tool_calls if c["tool"] == "search_eumetsat_products"
            and c["step"] == "r0.propose.thermal_specialist"]
    assert [c["status"] for c in leaf] == ["ok", "ok", "ok", "blocked", "blocked"]
    assert "Sub-agent tool limit (3" in leaf[3]["reason"]


def test_revision_tool_calls_are_capped(tmp_path):
    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "revise":
            done = sum(1 for c in req.contents for p in (c.parts or []) if p.function_response)
            if done < 3:
                return _call("eumetsat_agent", {"request": f"FCI over Chile, query {done}"})
        return default_script(role, phase, req)
    _, _, ledger, *_ = _run(tmp_path, script=script, max_rounds=1, max_revise_tool_calls=2)
    calls = [c for c in ledger.tool_calls if c["step"] == "r1.revise.thermal_specialist"
             and c["via"] is None]
    assert [c["status"] for c in calls][-1] == "blocked"
    assert sum(c["status"] != "blocked" for c in calls) == 2
    assert "Revision tool limit (2 calls)" in calls[-1]["reason"]


def test_pi_chooses_the_sub_agents_specialists_see(tmp_path):
    tools_seen = {}

    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            assert "SUB-AGENT ROSTER" in _request_text(req) and "fake gee_agent" in _request_text(req)
            return _text({**PROBLEM, "sub_agents": ["eumetsat_agent", "not_an_agent"]})
        if role in ("thermal_specialist", "optical_specialist") and phase == "propose":
            tools_seen[role] = _tool_names(req)
            if role == "thermal_specialist":
                assert "gee_agent" not in _request_text(req).split("calling these")[1].split("\n")[0]
        return default_script(role, phase, req)
    _, record, *_ = _run(tmp_path, script=script, max_rounds=0)
    assert record.problem.sub_agents == ["eumetsat_agent"]          # unknown names dropped
    assert "eumetsat_agent" in tools_seen["thermal_specialist"]
    assert "gee_agent" not in tools_seen["thermal_specialist"]       # offered before, not now
    assert not {"gee_agent", "cdse_agent"} & tools_seen["optical_specialist"]


def test_all_sub_agents_are_offered_when_the_pi_names_none(tmp_path):
    tools_seen = {}

    def script(role, phase, req):
        if role == "thermal_specialist" and phase == "propose":
            tools_seen[role] = _tool_names(req)
        return default_script(role, phase, req)
    _, record, *_ = _run(tmp_path, script=script, max_rounds=0)
    assert set(record.problem.sub_agents) == {"eumetsat_agent", "cdse_agent", "gee_agent"}
    assert {"eumetsat_agent", "gee_agent"} <= tools_seen["thermal_specialist"]


def test_specialists_search_the_catalogue_directly_and_hit_the_inventory_cache(tmp_path):
    _FakeCatalog.calls = []
    tools_seen = {}

    def script(role, phase, req):
        if role == "optical_specialist" and phase == "propose":
            tools_seen[role] = _tool_names(req)
            if not _has_tool_result(req):
                return _call("search_cdse_catalog", {
                    "collection": "Sentinel-2", "bbox": [-72.73, -37.41, -72.59, -37.22],
                    "start_date": "2026-01-15", "end_date": "2026-01-20", "product_type": "L2A"})
        return default_script(role, phase, req)
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(cdse=_FakeCatalog)))
    assert "search_cdse_catalog" in tools_seen["optical_specialist"]
    assert "cdse_agent" not in tools_seen["optical_specialist"]       # direct tool replaces it
    [call] = [c for c in ledger.tool_calls if c["step"] == "r0.propose.optical_specialist"]
    inventory_s2 = next(c for c in ledger.tool_calls
                        if c.get("inventory") and c["args"].get("collection") == "SENTINEL-2")
    assert call["cached_from"] == inventory_s2["id"]                  # narrower window, from cache
    assert call["catalog"]["dates"] == ["2026-01-17"]
    assert len(_FakeCatalog.calls) == 10      # no extra network call (8 + 2 Sentinel-3 retries)


def test_round_two_runs_only_while_issues_stay_open():
    from agenticeoecosystem.agents.virtual_lab.coordinator import VirtualLabCoordinator
    from agenticeoecosystem.agents.virtual_lab.schemas import CritiqueIssue, ScientificCritique

    def critique(severity):
        return ScientificCritique(author="x", target="y", round=1, issues=[CritiqueIssue(
            category="methodology", severity=severity, claim="c", rationale="r")])
    needs = VirtualLabCoordinator._needs_another_round
    assert needs([critique("blocking")], [])                 # blocking critique: re-review
    assert needs([critique("major")], [critique("major")])   # validator still unhappy
    assert not needs([critique("major")], [critique("minor")])
    assert not needs([], [])


def test_round_two_is_skipped_when_revisions_settle_the_issues(tmp_path, monkeypatch):
    from agenticeoecosystem.agents.virtual_lab.coordinator import VirtualLabCoordinator
    monkeypatch.setattr(VirtualLabCoordinator, "_needs_another_round", staticmethod(lambda *a: False))
    _, record, *_ = _run(tmp_path, max_rounds=2)
    assert record.rounds_run == 1 and record.termination == "settled_after_revision"
    _, record, *_ = _run(tmp_path, max_rounds=2, next_round_only_if_open=False)
    assert record.rounds_run == 2



class _FakeEumetsat:
    @staticmethod
    async def list_collections(query=None):
        return {"total": 3, "matching": 3, "collections": [
            {"id": "EO:EUM:DAT:0682", "title": "Active Fire Monitoring (netCDF) - MTG - 0 degree"},
            {"id": "EO:EUM:DAT:0801", "title": "Active Fire Monitoring (CAP) - MTG - 0 degree"},
            {"id": "EO:EUM:DAT:0962", "title": "SEVIRI Rapid Scan Fire Climate Data Record Release 1"}]}

    @staticmethod
    async def search_products(collection_id, bbox, start_date, end_date):
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 10)]
        return {"collection": collection_id, "start_date": start_date, "end_date": end_date,
                "total": 288, "listed": 288, "days": {"2026-01-17": times, "2026-01-18": times}}


class _FakeEarthdata:
    @staticmethod
    async def search_granules(short_name, bbox, start_date, end_date):
        return {"short_name": short_name, "start_date": start_date, "end_date": end_date, "total": 3,
                "listed": 3, "days": {"2026-01-17": ["04:54", "18:54"], "2026-01-18": ["04:36"]}}

    keywords: list = []

    @classmethod
    async def search_collections(cls, keyword, bbox=None, start_date=None, end_date=None, limit=20):
        cls.keywords.append(keyword)
        if "fire" not in keyword.lower():
            return {"total": 0, "listed": 0, "collections": []}
        return {"total": 1, "listed": 1, "collections": [
            {"short_name": "VNP14IMG", "version": "2", "title": "VIIRS active fire detections 375 m",
             "abstract": "Thermal infrared active fire detections and fire radiative power"}]}


class _FakeCds:
    @staticmethod
    async def dataset_info(dataset_id, start_date=None, end_date=None):
        return {"dataset": dataset_id, "title": f"ERA5 {dataset_id}", "start": "1940-01-01",
                "end": "2026-11-03", "update_frequency": "Daily", "status": "available",
                "covers_period": True}

    @staticmethod
    async def list_datasets(start_date=None, end_date=None):
        return {"total": 1, "matching": 1, "collections": [
            {"id": "reanalysis-era5-land", "title": "ERA5-Land hourly data",
             "description": "Reanalysis of land variables: temperature, wind, soil moisture",
             "keywords": ["Product type: Reanalysis"], "start": "1950-01-01", "end": "2026-11-03"}]}


def _all_catalogs():
    return Catalogs(cdse=_FakeCatalog, eumetsat=_FakeEumetsat, earthdata=_FakeEarthdata, cds=_FakeCds)


FIRE_NEEDS = ["thermal infrared active fire detections with fire radiative power",
              "burned area from satellite observations"]


def test_inventory_discovers_data_in_every_provider_without_using_the_budget(tmp_path):
    _FakeCatalog.calls, _FakeEarthdata.keywords = [], []
    tools_seen = {}

    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            return _text({**PROBLEM, "data_needs": FIRE_NEEDS})
        if role in ("thermal_specialist", "optical_specialist") and phase == "propose":
            tools_seen[role] = _tool_names(req)
        return default_script(role, phase, req)
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0, max_tool_calls=3)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=_all_catalogs()))
    assert record.problem.data_needs == FIRE_NEEDS
    inventory = [c for c in ledger.tool_calls if c.get("inventory")]
    tools = [c["tool"] for c in inventory]
    # Every provider listed, then the shortlist verified in the bbox and period.
    for listing in ("list_stac_collections", "list_eumetsat_collections", "list_cds_datasets",
                    "search_earthdata_collections"):
        assert listing in tools, listing
    assert _FakeEarthdata.keywords[:1] == [FIRE_NEEDS[0]]
    verified = {c["args"].get("collection_id") or c["args"].get("short_name") or c["args"].get("collection")
                for c in inventory if c["tool"] in ("count_eumetsat_products", "search_earthdata_granules",
                                                    "search_stac_items")}
    assert {"EO:EUM:DAT:0682", "VNP14IMG", "clms_burntarea_global_300m_daily_v4"} <= verified
    assert "EO:EUM:DAT:0801" not in verified              # format variant of 0682, counted once
    assert ledger.used_tool_calls() <= 3                   # inventory is free
    text = "\n".join(_requests(llm, "thermal_specialist", "propose"))
    assert "DATA DISCOVERY" in text and "VNP14IMG" in text and "288 items on 2 days" in text
    assert "UNUSABLE GEOMETRY" in text                     # MTG at 0 deg seen from Chile
    # One federated tool for every specialist: no per-provider routing.
    for role in ("thermal_specialist", "optical_specialist"):
        assert {"search_catalogues", "search_cdse_catalog"} <= tools_seen[role]
        assert not {"count_eumetsat_products", "search_earthdata_granules", "cds_dataset_info"} & tools_seen[role]


class _FailingEumetsat(_FakeEumetsat):
    @staticmethod
    async def list_collections(query=None):
        return {"error": "EUMETSAT collection listing failed: 503"}


class _NoListCds:
    """A CDS client without list_datasets: discovery cannot list it."""
    dataset_info = _FakeCds.dataset_info


def test_core_lookups_run_only_when_discovery_finds_nothing(tmp_path):
    llm = ScriptedLlm(script=default_script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0)
    _, _, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(eumetsat=_FailingEumetsat, cds=_NoListCds)))
    tools = [c["tool"] for c in ledger.tool_calls if c.get("inventory")]
    assert tools.count("count_eumetsat_products") == 3 and tools.count("cds_dataset_info") == 2
    text = "\n".join(_requests(llm, "thermal_specialist", "propose"))
    assert "listing FAILED" in text and "discovery found nothing" in text


class _GeocodingCatalog(_FakeCatalog):
    @classmethod
    async def geocode(cls, place_name, limit=3):
        return {"place_name": place_name, "results": [{"display_name": place_name,
                                                       "bbox": [-72.7, -37.4, -72.6, -37.2]}]}


def test_pi_geocodes_and_decides_without_sub_agents(tmp_path):
    tools_seen = {}

    def script(role, phase, req):
        if role == "lab_pi":
            tools_seen.setdefault(phase, _tool_names(req))
            if phase == "decide" and (n := _function_responses(req)) < 2:
                return _call("search_cdse_catalog", {"collection": "SENTINEL-2", "bbox": [-72.7, -37.4, -72.6, -37.2],
                                                     "start_date": "2026-01-0" + str(n + 1),
                                                     "end_date": "2026-01-0" + str(n + 1)})
        return default_script(role, phase, req)

    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(cdse=_GeocodingCatalog)))

    assert tools_seen["formulate"] == {"geocode_place"}                       # no cdse_agent
    assert "search_cdse_catalog" in tools_seen["decide"] and "cdse_agent" not in tools_seen["decide"]
    step = [c for c in ledger.tool_calls if c["step"] == "final.decide" and c["via"] is None]
    assert sum(c["status"] != "blocked" for c in step) == 1
    assert any(c["status"] == "blocked" and "Decision tool limit (1 calls)" in c["reason"] for c in step)
    assert "geocode_place" in "\n".join(_requests(llm, "lab_pi", "formulate"))


_GEOCODED = [-72.7, -37.4, -72.6, -37.2]   # what _GeocodingCatalog returns
_ELSEWHERE_BBOX = [10.0, 45.0, 10.1, 45.1]  # does not overlap the study area


class _LocatingCatalog(_GeocodingCatalog):
    """Accepts the agents' search_cdse_catalog call as well as the inventory's."""
    @classmethod
    async def search_scenes(cls, collection, bbox, start_date, end_date, *_, **__):
        return {"collection": collection, "bbox": bbox, "total": 1, "listed": 1,
                "products": [{"name": f"{collection}_X", "start": f"{start_date}T10:00:00Z"}]}


def _geocode_run(tmp_path, pi_bbox):
    """PI formulates with `pi_bbox`; the optical specialist searches somewhere else."""
    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            return _text({**PROBLEM, "study_area": {"name": "Biobio, Chile", "bbox": pi_bbox}})
        if role == "optical_specialist" and phase == "propose" and not _has_tool_result(req):
            return _call("search_cdse_catalog", {"collection": "SENTINEL-2", "bbox": _ELSEWHERE_BBOX,
                                                 "start_date": "2026-01-17", "end_date": "2026-01-18"})
        return default_script(role, phase, req)

    _FakeCatalog.calls = []
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0)
    _, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(cdse=_LocatingCatalog)))
    return record, ledger


def _validator_issues(ledger, agent):
    return [i for e in ledger.entries if e["kind"] == "validation"
            and e["critique"]["target"] == agent for i in e["critique"]["issues"]]


def test_unlocated_study_area_is_geocoded_by_the_lab(tmp_path):
    record, ledger = _geocode_run(tmp_path, pi_bbox=None)

    assert record.problem.study_area.bbox == _GEOCODED
    geo = [c for c in ledger.tool_calls if c["tool"] == "geocode_place" and c["step"] == "r0.geocode"]
    assert len(geo) == 1 and geo[0]["status"] == "ok" and geo[0]["inventory"]
    assert "set from geocoding" in geo[0]["note"]
    # The inventory now searches scenes in the geocoded area (it skipped them without a bbox).
    scenes = [c for c in ledger.tool_calls if c["step"] == "r0.inventory" and c["tool"] == "search_cdse_catalog"]
    assert scenes and all(c["args"]["bbox"] == _GEOCODED for c in scenes)


def test_pi_bbox_is_kept_and_off_area_searches_are_flagged(tmp_path):
    pi_bbox = [-72.73, -37.41, -72.59, -37.22]
    record, ledger = _geocode_run(tmp_path, pi_bbox=pi_bbox)

    assert record.problem.study_area.bbox == pi_bbox
    issues = _validator_issues(ledger, "optical_specialist")
    off = [i for i in issues if i["category"] == "coverage" and "does not overlap the study area" in i["rationale"]]
    assert len(off) == 1 and off[0]["severity"] == "major" and str(_ELSEWHERE_BBOX) in off[0]["claim"]
    assert not any("does not overlap" in i["rationale"] for i in _validator_issues(ledger, "thermal_specialist"))


def _run_execute(tmp_path, client, script=default_script):
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0, mode="execute")
    return (*asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        analysis_client=client)), llm)


def test_execute_mode_runs_the_analyses_and_concludes_from_results(tmp_path):
    from _virtual_lab_fakes import FakeSentinelHub

    def script(role, phase, req):
        if role == "experiment_designer" and phase == "design":
            assert "LAB ANALYSES" in _request_text(req)
        if role == "lab_pi" and phase == "conclude":
            text = _request_text(req)
            assert "EXPERIMENT RESULTS" in text and "res-4" in text
            return _text({"status": "decided", "summary": "Both sensors map the burn; S2 had clear scenes.",
                          "justification": [{"claim": "S2 and S1 agree (IoU > 0.85)", "kind": "verified",
                                             "result_id": "res-4"}]})
        return default_script(role, phase, req)

    client = FakeSentinelHub()
    answer, record, ledger, _, llm = _run_execute(tmp_path, client, script)
    assert [r.analysis for r in record.results] == ["optical_conditions", "s2_dnbr_burned_area",
                                                     "s1_backscatter_change", "s1_s2_agreement"]
    assert all(r.status == "ok" and r.tool_call_ids for r in record.results)
    assert record.decision.status.value == "decided"
    [claim] = record.decision.justification
    assert claim.kind.value == "verified" and claim.tool_call_ids == record.results[3].tool_call_ids
    steps = [e["step"] for e in ledger.entries if e["kind"] == "decision"]
    assert steps == ["final.decide", "final.conclude"]                    # before and after the data
    assert "Experiment results (executed)" in answer
    analysis_calls = [c for c in ledger.tool_calls if c.get("analysis")]
    assert len(analysis_calls) == 4 and ledger.used_tool_calls() == ledger.used_tool_calls()


def test_execute_mode_without_data_cannot_claim_a_decision(tmp_path):
    from _virtual_lab_fakes import FakeSentinelHub
    _, record, ledger, _, _ = _run_execute(tmp_path, FakeSentinelHub(fail=True))
    assert record.results and all(r.status == "failed" for r in record.results)
    assert record.decision.status.value == "insufficient_evidence"
    assert "no analysis ran successfully" in record.decision.status_note


def test_lab_agents_stay_read_only_in_execute_mode(tmp_path):
    from _virtual_lab_fakes import FakeSentinelHub

    def script(role, phase, req):
        if role == "cdse_agent" and not _has_tool_result(req):
            return _call("fetch_sentinel2", {"bbox": [0, 0, 1, 1]})
        return default_script(role, phase, req)

    _, _, ledger, _, _ = _run_execute(tmp_path, FakeSentinelHub(), script)
    assert not any(c["tool"] == "fetch_sentinel2" and c["status"] == "ok" for c in ledger.tool_calls)


def test_dataset_choices_are_checked_against_the_registry_and_reported(tmp_path):
    """A specialist that cites the inventory row of a discovered collection is grounded; one
    that names data no catalogue result shows is flagged. The final answer lists the data
    decision with its registry ids, and the record carries the grounding metrics."""
    import re

    from _virtual_lab_fakes import THERMAL_V0

    _FakeCatalog.calls, _FakeEarthdata.keywords = [], []

    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            return _text({**PROBLEM, "data_needs": FIRE_NEEDS})
        if role == "thermal_specialist" and phase == "propose":
            text = _request_text(req)
            viirs = re.search(r"\[(tc-\d{4})\] NASA Earthdata \(CMR\) · VNP14IMG", text).group(1)
            return _text({**THERMAL_V0, "datasets": [
                {"name": "VIIRS VNP14IMG", "role": "primary", "observable": "active fire detection",
                 "decision": "selected", "source_ids": [viirs]},
                {"name": "Imaginary Fire Product", "role": "reference", "observable": "fire perimeter"},
                {"name": "MTG FCI", "role": "baseline", "decision": "rejected",
                 "justification": "unusable geometry over Chile"}]})
        return default_script(role, phase, req)
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=0, max_tool_calls=3)
    answer, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=_all_catalogs()))
    assert "DATASET GROUNDING" in "\n".join(_requests(llm, "thermal_specialist", "propose"))
    check = next(e["critique"] for e in ledger.entries
                 if e["kind"] == "validation" and e["step"] == "r0.propose.thermal_specialist")
    flagged = {i["claim"] for i in check["issues"] if "catalogue result" in i["rationale"]}
    assert "Imaginary Fire Product" in flagged and "VIIRS VNP14IMG" not in flagged
    assert "MTG FCI" not in {i["claim"] for i in check["issues"]}   # rejected with a reason
    assert "## Data" in answer
    assert record.data_grounding["registry_entries"] > 0
    assert record.data_grounding["shortlisted_with_data"] >= 1
