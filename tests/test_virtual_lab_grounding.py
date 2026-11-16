"""Dataset grounding: the registry built from the ledger, the deterministic checks that tie
dataset choices to catalogue results, and the grounding metrics (no LLM, no network).
Run:
    uv run pytest tests/test_virtual_lab_grounding.py

The registry mirrors the Kypseli run of 2026-11-13: discovery found Sentinel-1 SLC bursts
(2023-2025) and NISAR GSLC scenes (from 2025-10) over the study area, yet the final plan
named neither. Expected selections are NOT hard-coded: the tests check that choices are tied to
the evidence, whatever they are.
"""
from _virtual_lab_fakes import PROBLEM, THERMAL_V0

from agenticeoecosystem.agents.virtual_lab.inventory import (
    data_registry,
    format_registry,
    resolve,
)
from agenticeoecosystem.agents.virtual_lab.ledger import data_records, with_registry_ids
from agenticeoecosystem.agents.virtual_lab.schemas import (
    DatasetChoice,
    ScientificProblem,
    ScientificProposal,
)
from agenticeoecosystem.agents.virtual_lab.validation import (
    check_grounding,
    grounding_metrics,
    validate_proposal,
)

problem = ScientificProblem.model_validate(PROBLEM)
BBOX = [23.72, 37.98, 23.76, 38.02]

BURSTS = {"total": 1340, "listed": 500, "days": {"2023-01-04": 4, "2023-01-09": 4, "2025-01-24": 4}}
NISAR = {"total": 37, "listed": 37, "days": {"2025-10-28": 2, "2026-06-18": 2, "2026-10-16": 2}}
SEARCH_RESULT = {"ranking": "embedding", "listings": [], "collections": [
    {"provider": "NASA Earthdata (CMR)", "id": "OPERA_L3_DISP-S1_V1", "title": "OPERA displacement",
     "match": 0.8, "status": "verified", "availability": "120 items on 60 days"},
    {"provider": "NASA Earthdata (CMR)", "id": "HRO", "title": "USGS orthoimagery",
     "match": 0.6, "status": "absent", "availability": "NONE found in the bbox and period"}]}


def _call(cid: str, tool: str, args: dict, result, **extra) -> dict:
    return {"id": cid, "tool": tool, "args": args, "status": "ok", "step": extra.pop("step", "r0.inventory"),
            "data": data_records(tool, args, result), **extra}


def _kypseli_calls() -> list[dict]:
    return [
        _call("tc-0001", "search_earthdata_granules", {"short_name": "SENTINEL-1_BURSTS", "bbox": BBOX},
              BURSTS, inventory=True),
        _call("tc-0002", "search_earthdata_granules", {"short_name": "NISAR_L2_GSLC_V1", "bbox": BBOX},
              NISAR, inventory=True),
        _call("tc-0003", "search_earthdata_granules", {"short_name": "HRO", "bbox": BBOX},
              {"total": 0, "listed": 0, "days": {}}, inventory=True),
        _call("tc-0004", "cds_dataset_info", {"dataset_id": "reanalysis-cerra-single-levels"},
              {"error": "CDS catalogue lookup failed: timeout"}, inventory=True),
        _call("tc-0005", "search_catalogues", {"query": "interferometric displacement"},
              SEARCH_RESULT, agent="sar_specialist", step="r0.propose.sar_specialist"),
        {"id": "tc-0006", "tool": "search_papers", "args": {}, "status": "ok", "step": "r1.literature"},
    ]


def _ds(**kw) -> DatasetChoice:
    return DatasetChoice.model_validate({"name": "Sentinel-1 SLC", "role": "primary", **kw})


def _severities(issues) -> list[str]:
    return sorted(i.severity for i in issues)


# ---------------------------------------------------------------------------- registry

def test_registry_keeps_every_discovered_candidate_with_its_status():
    reg = data_registry(_kypseli_calls())
    assert reg["tc-0001"].collection == "SENTINEL-1_BURSTS" and reg["tc-0001"].status == "verified"
    assert reg["tc-0002"].collection == "NISAR_L2_GSLC_V1" and "2025-10-28..2026-10-16" in reg["tc-0002"].availability
    assert reg["tc-0003"].status == "absent"
    assert reg["tc-0004"].status == "unknown"            # failed lookup: unknown, never absent
    assert reg["tc-0005/OPERA_L3_DISP-S1_V1"].found_by == "sar_specialist"
    assert "tc-0006" not in reg                           # not a data lookup
    assert [e.id for e in resolve(reg, "tc-0005")] == ["tc-0005/OPERA_L3_DISP-S1_V1", "tc-0005/HRO"]


def test_registry_section_shows_agent_searches_only_and_absent_ones_by_id():
    text = format_registry(data_registry(_kypseli_calls()))
    assert "[tc-0005/OPERA_L3_DISP-S1_V1]" in text and "searched by sar_specialist" in text
    assert "tc-0001" not in text                          # inventory rows are already in the prompt
    assert "NONE found in the bbox and period: tc-0005/HRO" in text


def test_tool_results_carry_the_registry_ids_to_cite():
    tagged = with_registry_ids("search_catalogues", SEARCH_RESULT, "tc-0042")
    assert [c["registry_id"] for c in tagged["collections"]] == ["tc-0042/OPERA_L3_DISP-S1_V1", "tc-0042/HRO"]
    assert with_registry_ids("search_cdse_catalog", {"total": 3}, "tc-0043")["registry_id"] == "tc-0043"
    assert with_registry_ids("search_papers", {"papers": []}, "tc-0044") == {"papers": []}


# ---------------------------------------------------------------------- schema parsing

def test_dataset_choice_reads_loose_ids_and_never_promotes_unknown_decisions():
    ds = _ds(source_ids="inventory:tc-0002, tc-0005/OPERA_L3_DISP-S1_V1", decision="Not selected")
    assert ds.source_ids == ["tc-0002", "tc-0005/OPERA_L3_DISP-S1_V1"] and ds.decision == "rejected"
    assert _ds(decision="maybe later").decision == "unverified"
    assert _ds().decision == "selected" and _ds().source_ids == []     # old outputs still parse


# ---------------------------------------------------------------------- grounding checks

def test_selected_dataset_must_cite_data_found_here():
    reg = data_registry(_kypseli_calls())
    assert check_grounding(_ds(source_ids=["tc-0001"]), reg) == []
    assert check_grounding(_ds(source_ids=["tc-0005/OPERA_L3_DISP-S1_V1"]), reg) == []
    assert _severities(check_grounding(_ds(), reg)) == ["major"]                     # no id
    assert _severities(check_grounding(_ds(role="ancillary"), reg)) == ["minor"]
    assert _severities(check_grounding(_ds(source_ids=["tc-0999"]), reg)) == ["major"]  # invented id
    assert _severities(check_grounding(_ds(source_ids=["tc-0003"]), reg)) == ["blocking"]  # none found


def test_incomplete_metadata_is_unknown_not_a_rejection():
    reg = data_registry(_kypseli_calls())
    issues = check_grounding(_ds(name="CERRA", role="ancillary", source_ids=["tc-0004"]), reg)
    assert _severities(issues) == ["minor"] and "unknown" in issues[0].rationale


def test_rejections_need_a_reason_and_unverified_options_stay_minor():
    reg = data_registry(_kypseli_calls())
    assert check_grounding(_ds(name="NISAR", decision="rejected", source_ids=["tc-0002"],
                               justification="archive starts 2025-10: no pre-tunnelling baseline"), reg) == []
    assert _severities(check_grounding(_ds(name="NISAR", decision="rejected"), reg)) == ["minor"]
    assert _severities(check_grounding(_ds(name="EGMS", decision="unverified"), reg)) == ["minor"]


def test_no_suitable_candidates_flags_selections_but_not_unverified_options():
    empty = data_registry([])
    assert _severities(check_grounding(_ds(), empty)) == ["major"]
    assert _severities(check_grounding(_ds(decision="unverified", role="ancillary"), empty)) == []


def test_validation_without_catalogue_access_skips_grounding():
    proposal = ScientificProposal.model_validate({**THERMAL_V0, "agent": "thermal_specialist"})
    plain = validate_proposal(proposal, problem, grounding_available=True)
    grounded = validate_proposal(proposal, problem, grounding_available=True,
                                 registry=data_registry(_kypseli_calls()))
    assert not any("catalogue result" in i.rationale for i in plain.issues)
    assert any("catalogue result" in i.rationale for i in grounded.issues)


def test_rejected_datasets_are_not_checked_as_inputs():
    proposal = ScientificProposal.model_validate({**THERMAL_V0, "agent": "thermal_specialist"})
    rejected = [d.model_copy(update={"decision": "rejected", "justification": "not needed"})
                for d in proposal.datasets]
    check = validate_proposal(proposal.model_copy(update={"datasets": rejected}), problem, True,
                              registry=data_registry(_kypseli_calls()))
    assert not any(i.category in ("dataset_availability", "coverage", "physical_inconsistency",
                                  "dataset_suitability") for i in check.issues)


# --------------------------------------------------------------------------- metrics

def test_grounding_metrics_measure_grounding_and_consideration():
    reg = data_registry(_kypseli_calls())
    choices = [_ds(source_ids=["tc-0001"]), _ds(name="Sentinel-2", role="ancillary"),
               _ds(name="NISAR", decision="rejected", source_ids=["tc-0002"], justification="too short"),
               _ds(name="EGMS", decision="unverified")]
    m = grounding_metrics(choices, reg, shortlisted=["tc-0001", "tc-0002", "tc-0003", "tc-0004"])
    assert (m["selected"], m["grounded"], m["grounding_rate"]) == (2, 1, 0.5)
    assert (m["rejected"], m["rejected_with_reason"], m["unverified"]) == (1, 1, 1)
    assert (m["shortlisted_with_data"], m["shortlisted_considered"]) == (2, 2)
    assert m["searched_by_agents"] == 2
