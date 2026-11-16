"""Generic ("how can one detect X anywhere?") questions: scoping, target size, failed lookups and
the seasonal inventory (no LLM, no network).
Run:
    uv run pytest tests/test_virtual_lab_exploratory.py

Regression case: the run of 2026-11-16 asked how to detect buried archaeology in an agricultural
field. The PI geocoded the words 'agricultural field' and got one 33 x 66 m field in Kerala; the
plan then used 100 m and 1 km thermal pixels on it without a finding, and two failed catalogue
searches were read as 'Sentinel-2 unavailable'.
"""
import asyncio
from typing import ClassVar

import pytest
from _virtual_lab_fakes import (
    PROBLEM,
    QUERY,
    ScriptedLlm,
    _call,
    _has_tool_result,
    _text,
    default_script,
    fake_existing_agents,
)

from agenticeoecosystem.agents.virtual_lab import coordinator
from agenticeoecosystem.agents.virtual_lab import inventory as inv
from agenticeoecosystem.agents.virtual_lab.config import VirtualLabConfig
from agenticeoecosystem.agents.virtual_lab.inventory import Catalogs
from agenticeoecosystem.agents.virtual_lab.runner import run_virtual_lab
from agenticeoecosystem.agents.virtual_lab.schemas import (
    DatasetChoice,
    EvidenceItem,
    ScientificProblem,
    is_named_place,
)
from agenticeoecosystem.agents.virtual_lab.validation import (
    area_extent_m,
    check_dataset,
    link_items,
)

KERALA_FIELD = [76.3755, 10.1208, 76.3758, 10.1214]          # the geocoded 'agricultural field'
TEST_SITE = [11.20, 42.40, 11.30, 42.48]                       # ~8 km across
ARCHAEOLOGY = {"feature": "buried walls and ditches", "min_m": 1, "max_m": 10}
GENERIC_PROBLEM = {
    **PROBLEM,
    "question": "How can buried archaeological features in agricultural fields be detected?",
    "scope": "generic",
    "study_area": {"name": "Ager Cosanus, Tuscany, Italy", "bbox": TEST_SITE,
                   "rationale": "cereal fields with crop marks documented in published surveys"},
    "time_period": {"start": "2023-01-01", "end": "2024-12-31"},
    "target": ARCHAEOLOGY,
}


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(coordinator, "INVENTORY_RETRY_DELAY_S", 0)


def _problem(**kw) -> ScientificProblem:
    return ScientificProblem.model_validate({**GENERIC_PROBLEM, **kw})


def _ds(name: str, **kw) -> DatasetChoice:
    return DatasetChoice.model_validate({"name": name, "role": "primary", **kw})


def _found(issues, category: str) -> list[str]:
    return [i.severity for i in issues if i.category == category]


# ------------------------------------------------------------------------------- scoping (A)

def test_kinds_of_place_are_not_place_names():
    assert not is_named_place("agricultural field") and not is_named_place("a dense city centre")
    assert is_named_place("Kypseli, Athens") and is_named_place("Αθήνα") and is_named_place("tile 33TUG")


def test_problem_reads_loose_scope_and_target():
    p = _problem(scope="Exploratory", target={"feature": "ditch", "min_m": 10, "max_m": 1})
    assert p.generic and (p.target.min_m, p.target.max_m) == (1, 10)       # swapped into order
    assert not _problem(scope=None).generic and not _problem(scope="Kypseli").generic
    assert _problem(target="about ten metres").target is None                # unusable: dropped
    assert _problem(target={"feature": "x", "max_m": -5}).target is None
    old = ScientificProblem.model_validate(PROBLEM)                           # earlier outputs parse
    assert old.scope == "site" and old.target is None and old.study_area.rationale == ""


# ---------------------------------------------------------------------------- pixel size (B)

def test_pixels_larger_than_the_target_are_a_major_finding():
    p = _problem(study_area={"name": "Thuruth field", "bbox": KERALA_FIELD})
    thermal = check_dataset(_ds("Landsat 8/9 TIRS", observable="land surface temperature anomalies"), p)
    assert "major" in _found(thermal, "dataset_suitability")
    assert any("~100 m" in i.rationale for i in thermal)                 # the thermal band, not 30 m
    assert "major" in _found(check_dataset(_ds("Sentinel-3 SLSTR", observable="thermal"), p),
                             "dataset_suitability")
    assert _found(check_dataset(_ds("Sentinel-2 MSI", observable="crop marks"), p),
                  "dataset_suitability") == ["minor"]                     # 10 m pixel, 10 m target


def test_the_study_area_must_be_wider_than_two_pixels():
    field = _problem(study_area={"name": "Thuruth field", "bbox": KERALA_FIELD}, target=None)
    assert round(area_extent_m(field)) == 33
    assert _found(check_dataset(_ds("Landsat 8/9 OLI", observable="NDVI"), field), "coverage") == ["minor"]
    assert _found(check_dataset(_ds("Landsat 8/9 OLI", observable="NDVI"), _problem(target=None)),
                  "coverage") == []


def test_sub_pixel_arguments_ancillary_data_and_large_targets_are_not_major():
    p = _problem()
    explained = _ds("Landsat 8/9 OLI", observable="linear crop marks spanning many pixels")
    assert _found(check_dataset(explained, p), "dataset_suitability") == ["minor"]
    assert _found(check_dataset(_ds("MODIS", role="ancillary", observable="LST"), p),
                  "dataset_suitability") == []
    kypseli = _problem(target={"feature": "settlement trough", "min_m": 20, "max_m": 50})
    assert _found(check_dataset(_ds("Sentinel-1 C-SAR", observable="InSAR"), kypseli),
                  "dataset_suitability") == []


# ---------------------------------------------------------------------- failed lookups (C)

def test_citing_a_failed_lookup_reads_as_unknown_not_as_a_missing_call():
    item = EvidenceItem(claim="Sentinel-2 L2A is unavailable", kind="retrieved", source="inventory:tc-0004")
    linked, issues = link_items([item], [], failed=frozenset({"tc-0004"}))
    assert linked[0].kind.value == "assumption"
    assert [i.severity for i in issues] == ["minor"] and "FAILED" in issues[0].rationale
    _, plain = link_items([item], [])
    assert "no matching tool call" in plain[0].rationale


class _FlakyCatalog:
    """The first Sentinel-2 search fails, the retry succeeds; scenes span two years."""
    calls: ClassVar[list] = []
    geocoded: ClassVar[list] = []
    DAYS = ("2023-01-05", "2023-06-10", "2023-06-22", "2024-06-15", "2024-12-20")

    @classmethod
    async def search_scenes(cls, collection, bbox, start_date, end_date, product_type=None, limit=20, **_):
        cls.calls.append(collection)
        if collection == "SENTINEL-2" and cls.calls.count("SENTINEL-2") == 1:
            return {"error": "CDSE catalogue search failed: ReadTimeout: "}
        products = [{"name": f"{collection}_{d}", "start": f"{d}T10:00:00Z", "cloudCover": c}
                    for d, c in zip(cls.DAYS, (80, 5, 10, 15, 90), strict=True)]
        return {"collection": collection, "bbox": bbox, "total": len(products), "listed": len(products),
                "products": products}

    @classmethod
    async def geocode(cls, place_name, limit=3):
        cls.geocoded.append(place_name)
        return {"place_name": place_name, "results": [{"display_name": place_name, "bbox": KERALA_FIELD}]}


# ----------------------------------------------------------------- seasonal inventory (D)

def test_long_or_generic_periods_are_summarised_per_calendar_month():
    s2 = inv.SCENE_SEARCHES[1]
    body = asyncio.run(_FlakyCatalog.search_scenes("SENTINEL-1", TEST_SITE, "2023-01-01", "2024-12-31"))
    row = inv.format_scene_row("tc-0003", s2, body, ("2023-01-01", "2024-12-31"), seasonal=True)
    assert "2023-01-05 to 2024-12-20" in row and "per calendar month over 2023-2024" in row
    assert "Jun 3d cloud~10% (3 clear)" in row and "Mar 0d" in row and "event" not in row
    event = _problem(scope="site", time_period={"start": "2026-01-17", "end": "2026-01-18"})
    assert not inv.seasonal_view(event, "2026-01-17", "2026-01-18")
    assert inv.seasonal_view(_problem(), "2026-01-17", "2026-01-18")                       # generic
    assert inv.seasonal_view(_problem(scope="site"), "2023-01-01", "2024-12-31")           # > 1 year


# ------------------------------------------------------------------------ whole meeting (A, E)

def _generic_run(tmp_path, formulated: dict):
    def script(role, phase, req):
        if role == "lab_pi" and phase == "formulate":
            if not _has_tool_result(req):
                return _call("geocode_place", {"place_name": "agricultural field"})
            return _text(formulated)
        return default_script(role, phase, req)

    _FlakyCatalog.calls, _FlakyCatalog.geocoded = [], []
    llm = ScriptedLlm(script=script, seen=[])
    config = VirtualLabConfig(participants=["thermal_specialist", "optical_specialist"],
                              literature_enabled=False, max_rounds=1)
    answer, record, ledger, _ = asyncio.run(run_virtual_lab(
        QUERY, config, llm=llm, existing_agents=fake_existing_agents(llm), artifact_dir=tmp_path,
        catalogs=Catalogs(cdse=_FlakyCatalog)))
    return answer, record, ledger, llm


def _prompts(llm, role, phase) -> str:
    return "\n".join(text for r, p, text in llm.seen if r == role and p == phase)


def test_generic_question_gets_a_reasoned_test_site_not_a_geocoded_noun(tmp_path):
    answer, record, ledger, llm = _generic_run(tmp_path, GENERIC_PROBLEM)

    pi_geo = next(c for c in ledger.tool_calls if c["tool"] == "geocode_place")
    assert "kind of place" in pi_geo["result_preview"]
    assert _FlakyCatalog.geocoded == ["Ager Cosanus, Tuscany, Italy"]   # only the named site
    assert record.problem.generic and record.problem.study_area.bbox == TEST_SITE
    assert "**Test site**" in answer and "crop marks documented" in answer
    assert "**Target size:** buried walls and ditches, 1-10 m." in answer

    # C: the failed Sentinel-2 search was retried inside one ledger entry, which succeeded.
    s2 = [c for c in ledger.tool_calls if c["step"] == "r0.inventory" and c["args"].get("collection") == "SENTINEL-2"]
    assert len(s2) == 1 and s2[0]["status"] == "ok" and _FlakyCatalog.calls.count("SENTINEL-2") == 2

    # D and E: participants see months, not an event; the critics and the PI get the generic rules.
    assert "per calendar month" in _prompts(llm, "thermal_specialist", "propose")
    assert "BEFORE the event" not in _prompts(llm, "thermal_specialist", "propose")
    assert "Ground truth:" in _prompts(llm, "scientific_critic", "critique")
    assert "Ground truth:" in _prompts(llm, "scientific_critic", "plan_review")
    assert "GENERIC" in _prompts(llm, "lab_pi", "decide") and "GENERIC" in _prompts(llm, "experiment_designer", "design")


def test_a_kind_of_place_without_coordinates_is_not_geocoded_by_the_lab(tmp_path):
    unnamed = {**GENERIC_PROBLEM, "study_area": {"name": "an agricultural field", "bbox": None}}
    answer, record, ledger, _ = _generic_run(tmp_path, unnamed)
    assert _FlakyCatalog.geocoded == [] and record.problem.study_area.bbox is None
    assert not [c for c in ledger.tool_calls if c["step"] == "r0.geocode"]
    assert "No reason was given for this site" in answer


def test_site_questions_keep_the_event_view_and_no_generic_rules(tmp_path):
    site = {**PROBLEM, "study_area": {"name": "EMSR859_1, Biobio, Chile", "bbox": TEST_SITE}}
    answer, record, _, llm = _generic_run(tmp_path, site)
    assert not record.problem.generic and "**Study area:**" in answer
    assert "DURING the event" in _prompts(llm, "thermal_specialist", "propose")
    assert "Ground truth:" not in _prompts(llm, "scientific_critic", "critique")
