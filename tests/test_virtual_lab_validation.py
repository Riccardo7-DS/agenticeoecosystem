"""Schemas, deterministic validation and status capping for the Virtual Lab (no LLM).

Run:
    uv run pytest tests/test_virtual_lab_validation.py
"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from _virtual_lab_fakes import OPTICAL_V0, PLAN, PROBLEM, THERMAL_V0, THERMAL_V1
from agenticeoecosystem.agents.virtual_lab.coordinator import _drop_checklist_echoes, parse_json_object
from agenticeoecosystem.agents.virtual_lab.literature import (
    arxiv_ids_in,
    broaden_query,
    is_duplicate_claim,
    is_relevant,
    limit_and,
    normalise_arxiv_id,
    query_terms,
    rank_from_payload,
)
from agenticeoecosystem.agents.virtual_lab.schemas import (
    DecisionStatus,
    Disagreement,
    EvidenceItem,
    EvidenceKind,
    CritiqueIssue,
    ExperimentPlan,
    LiteratureCheck,
    PaperRef,
    ScientificDecision,
    ScientificProblem,
    ScientificProposal,
    StudyArea,
)
from agenticeoecosystem.agents.virtual_lab.sensor_catalog import sensors_in_text
from agenticeoecosystem.agents.virtual_lab.validation import (
    check_dataset,
    finalize_status,
    link_evidence,
    link_items,
    literature_uptake,
    tool_call_evidence,
    validate_plan,
    validate_proposal,
)

problem = ScientificProblem.model_validate(PROBLEM)


def _proposal(data: dict, agent: str) -> ScientificProposal:
    return ScientificProposal.model_validate({**data, "agent": agent})


def test_verified_evidence_requires_an_experiment_result():
    with pytest.raises(ValidationError):
        EvidenceItem(claim="FCI detected first", kind=EvidenceKind.VERIFIED)
    assert EvidenceItem(claim="x", kind="verified", result_id="res-1").result_id == "res-1"


def test_bbox_is_validated():
    with pytest.raises(ValidationError):
        StudyArea(name="bad", bbox=[-72.6, -37.2, -72.7, -37.4])  # south > north
    with pytest.raises(ValidationError):
        StudyArea(name="bad", bbox=[0, 0, 1])


def test_geostationary_disk_edge_is_blocking_for_chile():
    issues = check_dataset(_proposal(THERMAL_V0, "t").datasets[0], problem)  # MTG FCI
    assert any(i.category == "coverage" and i.severity == "blocking" for i in issues)


def test_polar_thermal_sensor_has_no_coverage_issue():
    issues = check_dataset(_proposal(THERMAL_V1, "t").datasets[0], problem)  # VIIRS
    assert not [i for i in issues if i.severity in ("blocking", "major")]


def test_sensor_not_yet_operating_is_blocking():
    old = ScientificProblem.model_validate(
        {**PROBLEM, "time_period": {"start": "2010-01-01", "end": "2010-12-31"}})
    issues = check_dataset(_proposal(OPTICAL_V0, "o").datasets[0], old)  # Sentinel-2, launched 2015
    assert any(i.category == "dataset_availability" and i.severity == "blocking" for i in issues)


def test_physical_constraints_night_and_smoke():
    s2 = _proposal(OPTICAL_V0, "o").datasets[0]
    night = check_dataset(s2.model_copy(update={"observable": "night-time fire front"}), problem)
    assert any(i.category == "physical_inconsistency" and i.severity == "blocking" for i in night)
    smoke = check_dataset(s2.model_copy(update={"observable": "burned area under smoke"}), problem)
    assert any(i.category == "physical_inconsistency" for i in smoke)
    s1 = s2.model_copy(update={"name": "Sentinel-1 GRD", "observable": "burned area under smoke"})
    assert not any(i.category == "physical_inconsistency" for i in check_dataset(s1, problem))


def test_inaccessible_sensor_is_flagged():
    goes = _proposal(THERMAL_V0, "t").datasets[0].model_copy(update={"name": "GOES-16 ABI"})
    issues = check_dataset(goes, problem)
    assert any(i.category == "dataset_availability" and "No sub-agent" in i.rationale for i in issues)
    assert not any(i.category == "coverage" for i in issues)  # GOES-East sees Chile fine


def test_unbacked_retrieved_claim_is_downgraded():
    linked, issues = link_evidence(_proposal(OPTICAL_V0, "optical_specialist"), tool_calls=[])
    assert linked.evidence[0].kind == EvidenceKind.ASSUMPTION
    assert issues and issues[0].category == "unsupported_claim"


def test_retrieved_claim_links_to_matching_tool_call():
    calls = [{"id": "tc-0001", "tool": "eumetsat_agent", "via": None},
             {"id": "tc-0002", "tool": "search_eumetsat_products", "via": "eumetsat_agent"}]
    linked, issues = link_evidence(_proposal(THERMAL_V0, "thermal_specialist"), calls)
    assert linked.evidence[0].kind == EvidenceKind.RETRIEVED
    assert linked.evidence[0].tool_call_ids == ["tc-0001", "tc-0002"]
    assert not issues


def test_proposal_without_retrieved_evidence_is_insufficient():
    p, _ = link_evidence(_proposal(OPTICAL_V0, "optical_specialist"), [])
    crit = validate_proposal(p, problem, grounding_available=True)
    assert any(i.category == "evidence_sufficiency" and i.severity == "major" for i in crit.issues)


def test_plan_validation_flags_unknown_agents_and_missing_controls():
    plan = ExperimentPlan.model_validate({**PLAN, "reference_data": [], "baselines": [],
                                          "workflow": [{"order": 1, "description": "x",
                                                        "agent": "goes_agent"}]})
    crit = validate_plan(plan, problem, available_agents={"gee_agent", "eumetsat_agent"})
    rationales = " ".join(i.rationale for i in crit.issues)
    assert "goes_agent" in rationales
    assert "reference data" in rationales and "baseline" in rationales
    ok = validate_plan(ExperimentPlan.model_validate(PLAN), problem, {"gee_agent", "eumetsat_agent"})
    assert not [i for i in ok.issues if i.severity in ("blocking", "major")]


def _plan(**overrides) -> ExperimentPlan:
    return ExperimentPlan.model_validate({**PLAN, **overrides})


def _open(crit):
    return [i for i in crit.issues if i.severity in ("blocking", "major")]


AVAILABLE = {"gee_agent", "eumetsat_agent", "cdse_agent"}


def test_sensors_in_text_matches_whole_words_only():
    assert sensors_in_text("a necessary message about probability and policy") == []
    keys = {s.key for s in sensors_in_text("Sentinel-1 SAR backscatter vs MSG SEVIRI and Sentinel-2 dNBR")}
    assert keys == {"Sentinel-1 C-SAR", "MSG SEVIRI", "Sentinel-2 MSI"}


def test_reference_data_must_be_independent_of_evaluated_data():
    crit = validate_plan(_plan(reference_data=["VIIRS active fire data"]), problem, AVAILABLE)
    assert any(i.severity == "blocking" and "cannot validate itself" in i.rationale for i in crit.issues)
    ok = validate_plan(_plan(reference_data=["Copernicus EMS EMSR859 delineation"]), problem, AVAILABLE)
    assert not _open(ok)


def test_hypothesis_needs_a_linked_step_using_its_sensors():
    hyp = {"id": "thermal-H1", "statement": "MTG FCI detects the fire before VIIRS",
           "prediction": "FCI first", "falsification": "VIIRS first"}
    crit = validate_plan(_plan(hypotheses=[hyp]), problem, AVAILABLE)
    assert any("MTG FCI" in i.rationale and "cannot discriminate" in i.rationale for i in crit.issues)
    unlinked = [{**s, "hypotheses": []} for s in PLAN["workflow"]]
    crit = validate_plan(_plan(workflow=unlinked), problem, AVAILABLE)
    assert any("No workflow step is linked to thermal-H1" in i.rationale for i in crit.issues)


def test_step_agent_must_be_able_to_access_its_sensor():
    steps = [{"order": 1, "description": "search SLSTR fire product", "agent": "gee_agent",
              "hypotheses": ["thermal-H1"]}]
    crit = validate_plan(_plan(workflow=steps), problem, AVAILABLE)
    assert any("'gee_agent' cannot access Sentinel-3 SLSTR" in i.rationale for i in crit.issues)


def test_citation_ranking_puts_most_cited_first_and_unknown_last():
    ranked = rank_from_payload(["a1", "a2", "a3", "a4"], [
        {"citationCount": 5, "year": 2021}, {"citationCount": 50, "year": 2019}, None,
        {"citationCount": 20, "year": 2024}])["ranked"]
    assert [r["arxiv_id"] for r in ranked] == ["a2", "a4", "a1", "a3"]
    assert ranked[-1]["citation_count"] is None
    assert normalise_arxiv_id("http://arxiv.org/abs/2401.12345v2") == "2401.12345"


def test_arxiv_citation_links_only_to_the_search_that_returned_it():
    calls = [{"id": "tc-1", "tool": "search_papers", "status": "ok", "paper_ids": ["2101.00001"]},
             {"id": "tc-2", "tool": "search_papers", "status": "ok", "paper_ids": ["2202.00002"]}]
    items = [EvidenceItem(claim="smoke degrades dNBR", kind=EvidenceKind.RETRIEVED, source="arxiv:2101.00001"),
             EvidenceItem(claim="invented", kind=EvidenceKind.RETRIEVED, source="arxiv:9999.99999")]
    linked, issues = link_items(items, calls)
    assert linked[0].tool_call_ids == ["tc-1"]
    assert linked[1].kind == EvidenceKind.ASSUMPTION and len(issues) == 1


def test_sensor_scheduled_in_a_step_gets_coverage_checks():
    """FCI dropped from datasets but still fetched in a step (as in the 2026-11-01 run)."""
    steps = [*PLAN["workflow"], {"order": 3, "description": "Fetch fire detections from MTG FCI",
                                 "agent": "eumetsat_agent", "hypotheses": ["thermal-H1"]}]
    crit = validate_plan(_plan(workflow=steps), problem, AVAILABLE)
    assert any(i.category == "coverage" and i.severity == "blocking" and i.claim.startswith("step 3:")
               for i in crit.issues)


def test_revision_must_engage_with_literature_on_its_claims():
    check = LiteratureCheck(claim_id="r1-c1", claim="S2 dNBR under smoke", raised_by="thermal_specialist",
                            target="optical_specialist", verdict="contradicts",
                            papers=[PaperRef(arxiv_id="2101.00002")])
    ignoring = _proposal({**OPTICAL_V0, "revision": 1}, "optical_specialist")
    assert [i.severity for i in literature_uptake(ignoring, [check])] == ["minor"]
    citing = _proposal({**OPTICAL_V0, "revision": 1, "evidence": [
        {"claim": "smoke biases dNBR", "kind": "retrieved", "source": "arxiv:2101.00002"}]}, "optical_specialist")
    disputing = _proposal({**OPTICAL_V0, "revision": 1,
                           "limitations": ["arXiv:2101.00002 studied boreal fires, not Chile"]}, "optical_specialist")
    assert literature_uptake(citing, [check]) == [] == literature_uptake(disputing, [check])
    assert literature_uptake(_proposal(OPTICAL_V0, "optical_specialist"), [check]) == []  # round 0


def test_broaden_query_drops_quotes_and_ands():
    assert broaden_query('"MODIS FRP" AND ("orbit drift" OR gaps)') == "MODIS FRP OR (orbit drift OR gaps)"
    assert broaden_query("geostationary fire detection") is None


def test_reworded_claims_are_duplicates_and_distinct_claims_are_not():
    """The three MODIS rewordings and two distinct claims from the 2026-11-01 run."""
    modis = [
        "The thermal_specialist proposal includes MODIS as a dataset for fire detection and FRP, but MODIS "
        "is noted to have data continuity issues for recent dates due to Terra/Aqua orbit drift.",
        "MODIS is included as a dataset for fire detection despite its data continuity issues for recent "
        "dates due to Terra/Aqua orbit drift.",
        "MODIS is included as a dataset for fire detection and FRP despite potential data continuity "
        "issues for recent dates.",
    ]
    fci = "Geostationary sensors (MTG FCI, MSG SEVIRI) provide high-resolution thermal data suitable for fire detection."
    slstr = ("The proposal assumes that Sentinel-3 SLSTR provides thermal data suitable for detecting the "
             "fire's ignition time, but does not address the sensor's spatial resolution and revisit time.")
    assert is_duplicate_claim(modis[1], modis[:1]) and is_duplicate_claim(modis[2], modis[:1])
    assert not is_duplicate_claim(fci, modis) and not is_duplicate_claim(slstr, [fci, *modis])


def test_plan_review_issues_echoing_the_checklist_are_dropped():
    echo = CritiqueIssue(category="methodology", severity="major", rationale="r",
                         claim="Each hypothesis: do statement, prediction and falsification point the SAME way?")
    real = CritiqueIssue(category="methodology", severity="major", rationale="r",
                         claim="thermal-H1 predicts VIIRS first but is falsified if VIIRS is first")
    assert _drop_checklist_echoes([echo, real]) == [real]


def _decision(**kw) -> ScientificDecision:
    return ScientificDecision(status=kw.pop("status", DecisionStatus.DECIDED), summary="s", **kw)


def _cap(decision, **kw):
    args = dict(open_blocking=False, plan=ExperimentPlan.model_validate(PLAN), plan_blocking=False,
                n_valid_proposals=2, n_linked_evidence=1)
    return finalize_status(decision, **{**args, **kw})


def test_consensus_without_evidence_is_not_a_decision():
    capped = _cap(_decision(), n_linked_evidence=0)
    assert capped.status == DecisionStatus.NEEDS_EXPERIMENT
    assert "tool result" in capped.status_note


def test_disagreement_is_preserved_as_needs_experiment():
    d = _decision(disagreements=[Disagreement(topic="ignition sensor",
                                              positions={"a": "FCI", "b": "VIIRS"})])
    assert _cap(d).status == DecisionStatus.NEEDS_EXPERIMENT


def test_blocking_issue_without_plan_means_insufficient_evidence():
    assert _cap(_decision(), open_blocking=True, plan=None).status == DecisionStatus.INSUFFICIENT_EVIDENCE


def test_clean_decision_is_kept():
    d = _decision()
    assert _cap(d) is d


def test_needs_experiment_requires_a_usable_plan():
    d = _decision(status=DecisionStatus.NEEDS_EXPERIMENT)
    assert _cap(d, plan_blocking=True).status == DecisionStatus.INSUFFICIENT_EVIDENCE


def test_parse_json_object_tolerates_fences_and_trailing_commas():
    assert parse_json_object('```json\n{"a": 1,}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_relevance_filter_needs_two_query_terms():
    terms = query_terms(['"burned area" AND (smoke OR haze)', "ti:dNBR"])
    assert terms == {"burned", "area", "smoke", "haze", "dnbr"}
    assert is_relevant({"title": "Wildfire smoke", "abstract": "Burned area mapping under haze"}, terms)
    assert not is_relevant({"title": "Ambient backscatter", "abstract": "Smoke detectors in IoT"}, terms)
    assert is_relevant({"title": "x", "abstract": "dNBR"}, {"dnbr"})  # a single-term query needs one


def test_limit_and_keeps_one_top_level_and():
    assert limit_and("Sentinel-3 SLSTR AND fire detection AND ignition time") == (
        "Sentinel-3 SLSTR AND (fire detection OR ignition time)")
    assert limit_and("fire AND (MTG FCI OR SEVIRI) AND (resolution OR terrain)") == (
        "fire AND (MTG FCI OR SEVIRI OR resolution OR terrain)")
    assert limit_and('"burned area" AND (smoke OR haze)') == '"burned area" AND (smoke OR haze)'
    assert limit_and('"a AND b" AND c AND d') == '"a AND b" AND (c OR d)'  # quotes are opaque


def test_successful_calls_become_retrieved_evidence():
    calls = [
        {"id": "tc-1", "agent": "thermal_specialist", "tool": "eumetsat_agent", "via": None,
         "status": "ok", "result_preview": "Found  144 FCI\nproducts"},
        {"id": "tc-2", "agent": "eumetsat_agent", "tool": "search_eumetsat_products",
         "via": "eumetsat_agent", "status": "ok"},
        {"id": "tc-3", "agent": "thermal_specialist", "tool": "gee_agent", "via": None, "status": "error"},
        {"id": "tc-4", "agent": "thermal_specialist", "tool": "tle_agent", "via": None, "status": "ok",
         "result_preview": "passes"},
    ]
    cited = [EvidenceItem(claim="tle says", kind=EvidenceKind.RETRIEVED, source="tle_agent",
                          tool_call_ids=["tc-4"])]
    [item] = tool_call_evidence("thermal_specialist", calls, cited)
    assert item.claim == "eumetsat_agent returned: Found 144 FCI products"
    assert (item.kind, item.source, item.tool_call_ids) == (EvidenceKind.RETRIEVED, "eumetsat_agent",
                                                            ["tc-1", "tc-2"])


def test_designer_routes_analysis_to_gis_agent_when_loaded():
    from agenticeoecosystem.agents.virtual_lab import prompts
    with_gis = prompts.designer({}, "experiment_designer", {"gee_agent": "g", "gis_agent": "gis"})
    without = prompts.designer({}, "experiment_designer", {"gee_agent": "g"})
    assert "goes to gis_agent" in with_gis and "goes to gis_agent" not in without


def test_lab_literature_calls_link_only_by_arxiv_id():
    lab_search = {"id": "tc-9", "tool": "search_papers", "via": "arxiv_rag_agent", "status": "ok",
                  "lab_call": True, "paper_ids": ["2101.00001"]}
    by_name = EvidenceItem(claim="papers say so", kind=EvidenceKind.RETRIEVED, source="arxiv_rag_agent")
    by_id = EvidenceItem(claim="papers say so", kind=EvidenceKind.RETRIEVED, source="arxiv:2101.00001")
    (named, cited), _ = link_items([by_name, by_id], [lab_search])
    assert named.kind == EvidenceKind.ASSUMPTION  # never called arxiv_rag_agent itself
    assert cited.kind == EvidenceKind.RETRIEVED and cited.tool_call_ids == ["tc-9"]


def test_every_specialist_has_a_role_card_with_real_arxiv_ids():
    from agenticeoecosystem.agents.virtual_lab.participants import SPECIALISTS
    from agenticeoecosystem.agents.virtual_lab import prompts
    for spec in SPECIALISTS.values():
        assert spec.card, f"{spec.name} has no roles/{spec.name}.md"
    card = SPECIALISTS["foundation_model_specialist"].card
    assert {"2412.02732", "2311.00566", "2412.04204"} <= arxiv_ids_in(card)
    text = prompts.specialist({}, "sar_specialist", "SAR", [], 1, card=SPECIALISTS["sar_specialist"].card)
    assert "YOUR DOMAIN KNOWLEDGE" in text and "line-of-sight" in text


def test_dataset_role_synonyms_are_normalised():
    from agenticeoecosystem.agents.virtual_lab.schemas import DatasetChoice
    assert DatasetChoice(name="GNSS", role="validation").role == "reference"
    assert DatasetChoice(name="PANGAEA", role="Benchmark").role == "reference"
    assert DatasetChoice(name="SSL4EO-S12", role="pretraining").role == "primary"
    with pytest.raises(ValidationError):
        DatasetChoice(name="x", role="decorative")


@pytest.mark.parametrize("text,expected", [
    ("I currently do not have access to the tools or databases required to retrieve it.", True),
    ("Sentinel-2 data is not available for future dates.", True),
    ("I'm unable to fetch the Sentinel-2 data directly, but I can guide you.", True),
    ("Found 288 MTG FCI products for 2026-01-17.", False),
    ("I couldn't find any VIIRS datasets in the GEE catalog for that period.", False),
])
def test_no_answer_detection(text, expected):
    from agenticeoecosystem.agents.virtual_lab.ledger import is_no_answer
    assert is_no_answer(text) is expected


@pytest.mark.parametrize("result,expected", [
    ({"error": "boom"}, True),
    ({"isError": True, "content": []}, True),
    ({"content": [{"type": "text", "text": '{"error": "CDSE catalogue search failed: 400"}'}]}, True),
    ({"content": [{"type": "text", "text": '{"total": 12, "products": []}'}]}, False),
    ({"content": [{"type": "text", "text": "the error budget is fine"}]}, False),
    ("plain sub-agent reply", False),
])
def test_tool_errors_inside_mcp_replies_are_detected(result, expected):
    from agenticeoecosystem.agents.virtual_lab.ledger import reports_error
    assert reports_error(result) is expected


@pytest.mark.parametrize("name,expected", [
    ("Sentinel-3 SLSTR", ("SENTINEL-3", "SL_")),
    ("Sentinel-3 OLCI", ("SENTINEL-3", "OL_")),
    ("SENTINEL_1", ("SENTINEL-1", None)),
    ("Sentinel-2 MSI", ("SENTINEL-2", None)),
    ("sentinel-5p_l2", ("SENTINEL-5P", None)),
    ("LANDSAT-9", ("LANDSAT-9", None)),
    ("S1_GRD", ("SENTINEL-1", "GRD")),
    ("SENTINEL1_SLC", ("SENTINEL-1", "SLC")),
    ("SENTINEL2_L2A", ("SENTINEL-2", "MSIL2A")),
    ("S2", ("SENTINEL-2", None)),
    ("S3_SLSTR", ("SENTINEL-3", "SL_")),
])
def test_catalog_collection_names_are_normalised(name, expected):
    from agenticeoecosystem.servers.cdse_geocode.catalog import normalise_collection
    assert normalise_collection(name) == expected


@pytest.mark.parametrize("name,agent", [("MTG FCI", "eumetsat_agent"), ("MSG_SEVIRI", "eumetsat_agent"),
                                        ("VIIRS", "gee_agent"), ("MODIS", "gee_agent")])
def test_catalog_points_other_sensors_to_their_agent(name, agent):
    from agenticeoecosystem.servers.cdse_geocode.catalog import normalise_collection
    resolved, reason = normalise_collection(name)
    assert resolved is None and agent in reason and "does not mean" in reason


@pytest.mark.parametrize("text,expected", [
    ("The date range (2026-01-17 to 2026-01-18) is in the future, so no data exist.", True),
    ("as the dates are in the future.", True),
    ("Sentinel-2 data is not available for future dates.", True),
    ("Found 6 SLSTR FRP products for 2026-01-17.", False),
    ("Future missions such as CO2M will add coverage.", False),
])
def test_future_date_claims_are_detected(text, expected):
    from agenticeoecosystem.agents.virtual_lab.ledger import claims_future_dates
    assert claims_future_dates(text) is expected


def test_catalog_summary_keeps_dates_and_completeness():
    from agenticeoecosystem.agents.virtual_lab.ledger import catalog_summary
    body = {"collection": "SENTINEL-1", "product_type": "SLC", "start_date": "2026-01-10",
            "end_date": "2026-01-25", "total": 2, "listed": 2,
            "products": [{"start": "2026-01-16T10:04:00Z"}, {"start": "2026-01-22T10:05:00Z"}]}
    summary = catalog_summary({"content": [{"type": "text", "text": json.dumps(body)}]})
    assert summary == {"collection": "SENTINEL-1", "product_type": "SLC", "start": "2026-01-10",
                       "end": "2026-01-25", "complete": True, "dates": ["2026-01-16", "2026-01-22"]}
    assert catalog_summary({"content": [{"type": "text", "text": '{"error": "x"}'}]}) is None


def _dated_plan(description: str) -> ExperimentPlan:
    return ExperimentPlan.model_validate({**PLAN, "workflow": [
        {"order": 1, "description": description, "agent": "cdse_agent", "hypotheses": ["thermal-H1"]}]})


S1_SEARCH = {"collection": "SENTINEL-1", "product_type": None, "start": "2026-01-10",
             "end": "2026-01-25", "complete": True, "dates": ["2026-01-16", "2026-01-17", "2026-01-22"]}


def _date_issues(plan, catalog):
    check = validate_plan(plan, problem, {"cdse_agent", "gee_agent", "eumetsat_agent"}, catalog)
    return [i for i in check.issues if i.category == "dataset_availability" and "acquisition" in i.rationale]


def test_plan_step_on_a_date_without_acquisition_is_flagged():
    [issue] = _date_issues(_dated_plan("Fetch Sentinel-1 GRD for 2026-01-10 and 2026-01-18"),
                           [{**S1_SEARCH, "dates": S1_SEARCH["dates"] + ["2026-01-10"]}])
    assert "2026-01-18" in issue.claim and "2026-01-17" in issue.suggested_fix


def test_plan_dates_are_not_judged_without_a_complete_covering_search():
    plan = _dated_plan("Fetch Sentinel-1 GRD for 2026-01-18")
    assert not _date_issues(plan, [])                                     # never searched
    assert not _date_issues(plan, [{**S1_SEARCH, "complete": False}])     # list truncated
    assert not _date_issues(plan, [{**S1_SEARCH, "product_type": "SLC"}])  # other product searched
    assert not _date_issues(_dated_plan("Fetch Sentinel-1 GRD for 2026-01-17"), [S1_SEARCH])


# ---------------------------------------------------------------- data inventory

def _fake_get(payload):
    seen = []

    async def get(url, params):
        seen.append((url, params))
        if isinstance(payload, Exception):
            raise payload
        return payload
    return get, seen


ODATA_PAGE = {"@odata.count": 3, "value": [
    {"Name": "S2B_MSIL2A_20260117T143729_N0511_R096_T18HYF", "ContentDate": {"Start": "2026-01-17T14:37:29Z"},
     "Online": True, "Attributes": [{"Name": "cloudCover", "Value": 0.2}, {"Name": "productType", "Value": "S2MSI2A"}]},
    {"Name": "S2B_MSIL2A_20260117T143729_N0511_R096_T18HYG", "ContentDate": {"Start": "2026-01-17T14:37:29Z"},
     "Online": True, "Attributes": [{"Name": "cloudCover", "Value": 2.6}, {"Name": "productType", "Value": "S2MSI2A"}]},
    {"Name": "S2A_MSIL2A_20260122T143731_N0511_R096_T18HYF", "ContentDate": {"Start": "2026-01-22T14:37:31Z"},
     "Online": True, "Attributes": [{"Name": "cloudCover", "Value": 1.0}, {"Name": "productType", "Value": "S2MSI2A"}]},
]}


def test_scene_search_builds_the_filter_and_parses_products():
    import asyncio
    from agenticeoecosystem.servers.cdse_geocode.catalog import search_scenes
    get, seen = _fake_get(ODATA_PAGE)
    result = asyncio.run(search_scenes("SENTINEL2_L2A", [-72.73, -37.41, -72.59, -37.22],
                                       "2026-01-03T00:00", "2026-02-01", limit=5000, get=get))
    params = seen[0][1]
    assert "Collection/Name eq 'SENTINEL-2'" in params["$filter"]
    assert "contains(Name,'MSIL2A')" in params["$filter"] and params["$top"] == "1000"
    assert result["total"] == 3 and result["listed"] == 3 and result["start_date"] == "2026-01-03"
    assert result["products"][0]["cloudCover"] == 0.2


def test_scene_search_failure_is_an_error_not_an_empty_list():
    import asyncio
    from agenticeoecosystem.servers.cdse_geocode.catalog import search_scenes
    get, _ = _fake_get(RuntimeError("503"))
    result = asyncio.run(search_scenes("SENTINEL-1", [0, 0, 1, 1], "2026-01-01", "2026-01-02", get=get))
    assert "error" in result and "products" not in result


def test_openeo_collections_are_filtered_by_extent():
    import asyncio
    from agenticeoecosystem.servers.cdse_geocode.catalog import list_openeo_collections

    def coll(cid, bbox, interval):
        return {"id": cid, "title": cid.title(), "extent": {"spatial": {"bbox": [bbox]},
                                                            "temporal": {"interval": [interval]}}}
    get, _ = _fake_get({"collections": [
        coll("SENTINEL2_L2A", [-180, -90, 180, 90], ["2015-06-27T10:25:31Z", None]),
        coll("SENTINEL1_GRD", [0, 0, 0, 0], [None, None]),                       # unknown extent: kept
        coll("CLMS_SWI_EUROPE_1KM_DAILY_V1", [-11, 34, 45, 72], ["2014-12-31T12:00:01Z", None]),
        coll("ESA_WORLDCOVER_10M_2020_V1", [-180, -90, 180, 90], ["2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"]),
    ]})
    result = asyncio.run(list_openeo_collections([-72.73, -37.41, -72.59, -37.22], "2026-01-17",
                                                 "2026-01-18", get=get))
    assert [c["id"] for c in result["collections"]] == ["SENTINEL2_L2A", "SENTINEL1_GRD"]
    assert result["total"] == 4 and result["matching"] == 2


def test_inventory_rows_summarise_days_and_never_turn_failures_into_absence():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    s2 = inv.SCENE_SEARCHES[1]
    window = inv.search_window("2026-01-17", "2026-01-18", s2, 14)
    assert window == ("2026-01-03", "2026-02-01")
    assert inv.search_window("2026-01-17", "2026-01-18", inv.SCENE_SEARCHES[2], 14) == ("2026-01-16", "2026-01-19")
    body = {"total": 3, "listed": 3, "products": [
        {"start": p["ContentDate"]["Start"], **{a["Name"]: a["Value"] for a in p["Attributes"]}}
        for p in ODATA_PAGE["value"]]}
    row = inv.format_scene_row("tc-0003", s2, body, window)
    assert "[tc-0003]" in row and "3 scenes on 2 days" in row
    assert "2026-01-17 (2) S2MSI2A cloud 0-3% 14:37 UTC" in row
    assert "first 3 of 9" in inv.format_scene_row("tc-0003", s2, {**body, "total": 9}, window)
    failed = inv.format_scene_row("tc-0004", s2, {"error": "timeout"}, window)
    assert "SEARCH FAILED" in failed and "NOT absent" in failed
    assert "no scenes" in inv.format_scene_row("tc-0005", s2, {"total": 0, "listed": 0, "products": []}, window)
    assert not inv.valid_period("unknown", "2026-01-18") and inv.valid_period("2026-01-17", "2026-01-18")


def test_inventory_text_names_what_it_does_not_cover():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    collections = [{"id": "SENTINEL2_L2A", "title": "Sentinel-2 L2A"},
                   {"id": "CLMS_BA_GLOBAL_300M_DAILY_V4", "title": "CLMS Burnt Area (BA) Global 300m daily V4",
                    "keywords": ["Burnt Area", "fire disturbance"]},
                   {"id": "CLMS_SCE_GLOBAL_1KM_DAILY_V1", "title": "CLMS Snow Cover Extent (SCE) Global 1km daily V1",
                    "keywords": ["cryosphere"]}]
    text = inv.format_inventory("EMSR859", [-72.73, -37.41, -72.59, -37.22], ["- [tc-0002] row"],
                                {"total": 130, "matching": 3, "collections": collections},
                                "tc-0001", "Map the final burned area of the wildfire")
    assert "MTG, FCI, MSG, SEVIRI -> ask eumetsat_agent" in text and "VIIRS, MODIS -> ask gee_agent" in text
    assert "[tc-0001]" in text and "CLMS_BA_GLOBAL_300M_DAILY_V4" in text and "needs credits" in text
    assert "CLMS_SCE" not in text and "+1 more not related" in text


def _catalog_entry(call_id, args, products):
    body = {"collection": "SENTINEL-2", "product_type": "MSIL2A", "bbox": args["bbox"],
            "start_date": args["start_date"], "end_date": args["end_date"],
            "total": len(products), "listed": len(products), "products": products}
    return {"id": call_id, "tool": "search_cdse_catalog", "args": args}, body


def test_complete_catalogue_search_answers_narrower_repeats():
    from agenticeoecosystem.agents.virtual_lab.ledger import LabLedger
    bbox = [-72.73, -37.41, -72.59, -37.22]
    ledger = LabLedger()
    entry, body = _catalog_entry("tc-0003", {"collection": "SENTINEL-2", "bbox": bbox, "start_date": "2026-01-03",
                                             "end_date": "2026-02-01", "product_type": "MSIL2A", "limit": 500},
                                 [{"name": "S2B_MSIL2A_20260117", "start": "2026-01-17T14:37:29Z", "cloudCover": 0.2},
                                  {"name": "S2A_MSIL2A_20260112", "start": "2026-01-12T14:37:29Z", "cloudCover": 60.0}])
    ledger.remember(entry, body)
    hit_id, result = ledger.recall("search_cdse_catalog", {
        "collection": "sentinel 2", "bbox": bbox, "start_date": "2026-01-10", "end_date": "2026-01-20",
        "product_type": "S2MSI2A", "max_cloud_cover": 10})
    assert hit_id == "tc-0003" and [p["name"] for p in result["products"]] == ["S2B_MSIL2A_20260117"]
    assert result["total"] == 1 and result["start_date"] == "2026-01-10"
    # Outside the cached window, another bbox, or another product: not answered.
    assert ledger.recall("search_cdse_catalog", {"collection": "SENTINEL-2", "bbox": bbox,
                                                 "start_date": "2025-12-30", "end_date": "2026-01-20"}) is None
    assert ledger.recall("search_cdse_catalog", {"collection": "SENTINEL-2", "bbox": [0, 0, 1, 1],
                                                 "start_date": "2026-01-10", "end_date": "2026-01-20"}) is None
    assert ledger.recall("search_cdse_catalog", {"collection": "SENTINEL-2", "bbox": bbox, "product_type": "MSIL1C",
                                                 "start_date": "2026-01-10", "end_date": "2026-01-20"}) is None


def test_unfiltered_search_answers_a_product_filtered_one_and_incomplete_ones_are_not_reused():
    from agenticeoecosystem.agents.virtual_lab.ledger import LabLedger
    bbox = [-72.73, -37.41, -72.59, -37.22]
    ledger = LabLedger()
    args = {"collection": "SENTINEL-1", "bbox": bbox, "start_date": "2026-01-03", "end_date": "2026-02-01"}
    entry, body = _catalog_entry("tc-0002", args, [
        {"name": "S1A_IW_GRDH_1SDV_20260117", "start": "2026-01-17T23:35:00Z"},
        {"name": "S1A_IW_SLC__1SDV_20260117", "start": "2026-01-17T23:35:00Z"}])
    ledger.remember(entry, {**body, "collection": "SENTINEL-1", "product_type": None})
    _, result = ledger.recall("search_cdse_catalog", {**args, "collection": "S1_GRD", "start_date": "2026-01-17",
                                                      "end_date": "2026-01-17"})
    assert [p["name"] for p in result["products"]] == ["S1A_IW_GRDH_1SDV_20260117"]
    incomplete = LabLedger()
    incomplete.remember(entry, {**body, "total": 9})
    assert incomplete.recall("search_cdse_catalog", {**args, "start_date": "2026-01-17",
                                                     "end_date": "2026-01-17"}) is None
    # Sub-agent requests match ignoring case, spacing and the date prefix the guard adds.
    ledger.remember({"id": "tc-0009", "tool": "eumetsat_agent",
                     "args": {"request": "[Today is 2026-11-09; earlier dates are in the past.] FCI  over Chile"}},
                    "Found 144")
    assert ledger.recall("eumetsat_agent", {"request": "fci over chile"}) == ("tc-0009", "Found 144")


def test_inventory_citations_link_to_inventory_calls():
    from agenticeoecosystem.agents.virtual_lab.validation import link_items
    calls = [{"id": "tc-0001", "tool": "list_openeo_collections", "status": "ok", "lab_call": True, "inventory": True},
             {"id": "tc-0003", "tool": "search_cdse_catalog", "status": "ok", "lab_call": True, "inventory": True},
             {"id": "tc-0009", "tool": "search_papers", "status": "ok", "lab_call": True}]
    items = [EvidenceItem(claim="S2 on 2026-01-17 at 0.2% cloud", kind=EvidenceKind.RETRIEVED,
                          source="inventory:tc-0003"),
             EvidenceItem(claim="Several collections exist", kind=EvidenceKind.RETRIEVED, source="inventory"),
             EvidenceItem(claim="CDSE has S2", kind=EvidenceKind.RETRIEVED, source="search_cdse_catalog")]
    linked, issues = link_items(items, calls)
    assert linked[0].tool_call_ids == ["tc-0003"]
    assert linked[1].tool_call_ids == ["tc-0001", "tc-0003"]
    # Naming the tool does not show the agent read what the lab retrieved.
    assert linked[2].kind == EvidenceKind.ASSUMPTION and len(issues) == 1



def test_eumetsat_extras_are_domain_related_products_not_word_overlaps():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    collections = [
        {"id": "EO:EUM:DAT:0737", "title": "IRS Level 1 reduced sub-set of 300 Principal Components - MTG"},
        {"id": "EO:EUM:DAT:0682", "title": "Active Fire Monitoring (netCDF) - MTG - 0 degree"},
        {"id": "EO:EUM:DAT:0801", "title": "Active Fire Monitoring (CAP) - MTG - 0 degree"},
        {"id": "EO:EUM:DAT:0962", "title": "SEVIRI Rapid Scan Fire Climate Data Record Release 1"},
        {"id": "EO:EUM:DAT:0417", "title": "SLSTR Level 2 Fire Radiative Power - Sentinel-3"},
        {"id": "EO:EUM:DAT:0662", "title": "FCI Level 1c Normal Resolution Image Data - MTG - 0 degree"},
    ]
    extras = inv.eumetsat_extras(collections, "Compare the components that constrain ignition time "
                                              "and burned area of the wildfire")
    assert [e.key for e in extras] == ["EO:EUM:DAT:0682", "EO:EUM:DAT:0417"]


def test_days_rows_summarise_cadence_and_overpasses():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    ten_min = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 10)]
    row = inv.format_days_row("tc-0017", "MTG FCI", {"total": 144, "listed": 144,
                                                    "days": {"2026-01-17": ten_min}}, ("2026-01-17", "2026-01-17"))
    assert "2026-01-17 (144) 00:00-23:50 UTC, every ~10 min" in row
    row = inv.format_days_row("tc-0009", "VIIRS", {"total": 2, "listed": 2, "days": {"2026-01-17": ["06:12", "18:36"]}},
                              ("2026-01-17", "2026-01-17"), "granules")
    assert "2 granules" in row and "06:12 18:36 UTC" in row
    assert "NOT absent" in inv.format_days_row("tc-1", "x", {"error": "429"}, ("a", "b"))
    assert "covers the study period" in inv.format_cds_row("tc-2", {"title": "ERA5", "dataset": "e", "start": "1940-01-01",
                                                                    "end": "2026-11-03", "covers_period": True})
    assert "does NOT yet cover" in inv.format_cds_row("tc-2", {"title": "ERA5", "dataset": "e", "covers_period": False})


def test_narrower_product_searches_are_answered_from_a_complete_one():
    from agenticeoecosystem.agents.virtual_lab.ledger import LabLedger
    bbox = [-72.73, -37.41, -72.59, -37.22]
    ledger = LabLedger()
    args = {"short_name": "VJ114IMG", "bbox": bbox, "start_date": "2026-01-16", "end_date": "2026-01-19"}
    ledger.remember({"id": "tc-0010", "tool": "search_earthdata_granules", "args": args},
                    {"short_name": "VJ114IMG", "start_date": "2026-01-16", "end_date": "2026-01-19", "total": 3,
                     "listed": 3, "days": {"2026-01-16": ["05:12"], "2026-01-17": ["04:54", "18:54"]}})
    hit_id, result = ledger.recall("search_earthdata_granules", {**args, "short_name": "vj114img",
                                                                  "start_date": "2026-01-17", "end_date": "2026-01-17"})
    assert hit_id == "tc-0010" and result["total"] == 2 and list(result["days"]) == ["2026-01-17"]
    assert ledger.recall("search_earthdata_granules", {**args, "short_name": "MOD14"}) is None
    # A monthly granule that began before the window can't be placed in a narrower one.
    monthly = LabLedger()
    monthly.remember({"id": "tc-0014", "tool": "search_earthdata_granules", "args": {**args, "short_name": "MCD64A1"}},
                     {"start_date": "2026-01-16", "end_date": "2026-01-19", "total": 1, "listed": 1,
                      "days": {"2026-01-01": ["00:00"]}})
    assert monthly.recall("search_earthdata_granules", {**args, "short_name": "MCD64A1",
                                                        "start_date": "2026-01-17", "end_date": "2026-01-17"}) is None


def test_geostationary_rows_say_whether_the_area_is_usable():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    chile = [-72.73, -37.41, -72.59, -37.22]
    note = inv.geometry_note(chile, 0.0)
    assert note.startswith("UNUSABLE GEOMETRY") and "data exist" in note
    assert inv.geometry_note([10.0, 40.0, 11.0, 41.0], 0.0).startswith("Geometry OK")
    assert inv.geometry_note(chile, None) == ""
    assert "GOES-East ABI" in inv.best_geostationary(chile)
    assert inv.geostationary_lon("Active Fire Monitoring - MTG - 0 degree") == 0.0
    assert inv.geostationary_lon("High Rate SEVIRI Level 1.5 Image Data - MSG - Indian Ocean") == inv.IODC_LON
    assert inv.geostationary_lon("AVHRR Level 1B - Metop - Global") is None
    row = inv.format_days_row("tc-0017", "MTG FCI", {"total": 1, "listed": 1, "days": {"2026-01-17": ["00:00"]}},
                              ("2026-01-17", "2026-01-17"), geometry=note)
    assert row.endswith(note)
    assert "GEOMETRY" not in inv.format_days_row("tc-1", "x", {"total": 0}, ("a", "b"), geometry=note)


def test_norms_distinguish_not_found_unusable_and_not_checked():
    from agenticeoecosystem.agents.virtual_lab.prompts import SCIENTIFIC_NORMS
    assert all(k in SCIENTIFIC_NORMS for k in ("NOT FOUND", "UNUSABLE GEOMETRY", "NOT CHECKED"))


def test_scene_rows_separate_pre_and_post_event_days():
    from agenticeoecosystem.agents.virtual_lab import inventory as inv
    s1 = inv.SCENE_SEARCHES[0]
    body = {"total": 3, "listed": 3, "products": [
        {"start": "2026-01-10T10:05:00Z", "productType": "IW_GRDH_1S"},
        {"start": "2026-01-17T09:57:00Z", "productType": "IW_GRDH_1S"},
        {"start": "2026-01-22T10:05:00Z", "productType": "IW_GRDH_1S"}]}
    row = inv.format_scene_row("tc-0002", s1, body, ("2026-01-03", "2026-02-01"), ("2026-01-17", "2026-01-18"))
    assert "BEFORE the event: 2026-01-10" in row
    assert "DURING the event: 2026-01-17" in row
    assert "AFTER the event: 2026-01-22" in row
    only_after = {**body, "total": 1, "listed": 1, "products": body["products"][2:]}
    assert "BEFORE the event: none" in inv.format_scene_row(
        "tc-0002", s1, only_after, ("2026-01-03", "2026-02-01"), ("2026-01-17", "2026-01-18"))


def test_geocode_returns_bboxes_without_the_mcp_server():
    from agenticeoecosystem.servers.cdse_geocode.catalog import geocode

    async def get(url, params):
        assert params["q"] == "Concepcion"
        return [{"display_name": "Concepción, Chile", "boundingbox": ["-36.9", "-36.7", "-73.1", "-72.9"]}]

    async def fail(url, params):
        raise TimeoutError()

    result = asyncio.run(geocode("Concepcion", get=get))
    assert result["results"][0]["bbox"] == [-73.1, -36.9, -72.9, -36.7]       # west, south, east, north
    assert asyncio.run(geocode("x", get=fail))["error"] == "Geocoding failed: TimeoutError"


def test_plan_steps_that_exclude_a_sensor_are_not_flagged_for_it():
    from agenticeoecosystem.agents.virtual_lab.schemas import ExperimentStep
    from agenticeoecosystem.agents.virtual_lab.validation import sensors_used
    step = ExperimentStep(order=1, agent="gee_agent", description=(
        "Retrieve VIIRS and MODIS active fire data. Exclude geostationary sensors (MTG FCI, MSG "
        "SEVIRI) due to UNUSABLE GEOMETRY (viewing zenith 85.2 deg)."), inputs=["VIIRS VNP14IMG"])
    assert {s.key for s in sensors_used(step)} == {"VIIRS", "MODIS"}
    used = ExperimentStep(order=2, agent="eumetsat_agent", description="Fetch MTG FCI 10-min data.")
    assert [s.key for s in sensors_used(used)] == ["MTG FCI"]


def test_fixed_analyses_run_on_data_and_pick_the_clearest_dates(tmp_path):
    from _virtual_lab_fakes import FakeSentinelHub
    from agenticeoecosystem.agents.virtual_lab import analyses as A
    client = FakeSentinelHub()
    ctx = A.AnalysisContext([-72.73, -37.41, -72.59, -37.22], "2026-01-17", "2026-01-18", out_dir=tmp_path)
    out = {}
    for name in A.with_dependencies(["s1_s2_agreement"]):
        out[name] = asyncio.run(A.RUNNERS[name](ctx, client))
    assert list(out) == ["optical_conditions", "s2_dnbr_burned_area", "s1_backscatter_change", "s1_s2_agreement"]
    cond = out["optical_conditions"]
    assert cond["status"] == "ok" and cond["metrics"]["max_cloud_during"] == 0.6
    assert cond["details"]["pre_scene"]["date"] < "2026-01-17" < "2026-01-18" < cond["details"]["post_scene"]["date"]
    s2, s1, agree = out["s2_dnbr_burned_area"], out["s1_backscatter_change"], out["s1_s2_agreement"]
    total_km2 = ctx.grid()[0] * ctx.grid()[1] * ctx.grid()[2]
    assert abs(s2["metrics"]["burned_km2"] - total_km2 / 2) < 0.05 * total_km2   # the left half burned
    assert abs(s1["metrics"]["burned_km2"] - total_km2 / 2) < 0.08 * total_km2
    assert agree["metrics"]["iou"] > 0.85 and agree["artifacts"]


def test_conclusion_keeps_verified_only_for_successful_results():
    from agenticeoecosystem.agents.virtual_lab.schemas import ExperimentResult, ScientificDecision
    from agenticeoecosystem.agents.virtual_lab.validation import (finalize_conclusion, link_results,
                                                                  normalise_conclusion)
    data = normalise_conclusion({"status": "decided", "summary": "S2", "justification": [
        {"claim": "S2 usable 99%", "kind": "verified", "source": "result:res-2"},
        {"claim": "S1 failed", "kind": "verified", "result_id": "res-3"},
        {"claim": "no id", "kind": "verified"}]})
    results = [ExperimentResult(id="res-2", analysis="s2_dnbr_burned_area", tool_call_ids=["tc-0042"]),
               ExperimentResult(id="res-3", analysis="s1_backscatter_change", status="failed")]
    d = ScientificDecision.model_validate(data)
    linked = link_results(d.justification, results)
    assert [e.kind.value for e in linked] == ["verified", "assumption", "assumption"]
    assert linked[0].tool_call_ids == ["tc-0042"]
    kept = finalize_conclusion(d.model_copy(update={"justification": linked}), results)
    assert kept.status.value == "decided"
    capped = finalize_conclusion(d.model_copy(update={"justification": linked[1:]}), results)
    assert capped.status.value == "needs_experiment"
    failed = finalize_conclusion(d, [results[1]])
    assert failed.status.value == "insufficient_evidence"
