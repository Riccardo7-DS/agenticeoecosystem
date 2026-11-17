"""The Virtual Lab's tools as MCP servers (servers/cdse_analysis, servers/eo_research), offline.

cdse_analysis runs each analysis as its own call and keeps a study's maps on disk
(analyses.AnalysisStore), so later calls reuse them instead of downloading again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import ModuleType

import pytest
from _virtual_lab_fakes import FakeSentinelHub
from test_virtual_lab_discovery import BBOX as DISCOVERY_BBOX
from test_virtual_lab_discovery import FakeEmbedder, _catalogs

from agenticeoecosystem.agents.virtual_lab import analyses as an
from agenticeoecosystem.agents.virtual_lab import discovery as d

BBOX = [-72.70, -37.40, -72.60, -37.30]
START, END = "2026-01-16", "2026-01-18"   # the fake burns after 2026-01-18


def _stored(name: str, client: FakeSentinelHub, root: Path, refresh: bool = False) -> dict:
    ctx = an.AnalysisContext(bbox=BBOX, start=START, end=END)   # a fresh call
    return asyncio.run(an.AnalysisStore(root).run(name, ctx, client, refresh))


def _study(root: Path) -> Path:
    return root / an.AnalysisStore.study_key(BBOX, START, END, 14)


def test_an_unreadable_store_is_recomputed_not_an_error(tmp_path):
    _stored("optical_conditions", FakeSentinelHub(), tmp_path)
    (_study(tmp_path) / "results.json").write_text('{"optical_conditions": {"sta')   # write cut short
    again = _stored("optical_conditions", FakeSentinelHub(), tmp_path)
    assert again["status"] == "ok" and not again["reused"] and again["ran"] == ["optical_conditions"]


def test_separate_calls_share_maps_through_the_store(tmp_path):
    client, store = FakeSentinelHub(), tmp_path
    s2 = _stored("s2_dnbr_burned_area", client, store)
    assert s2["status"] == "ok" and s2["ran"] == ["optical_conditions", "s2_dnbr_burned_area"]
    assert s2["metrics"]["burned_km2"] > 0
    s1 = _stored("s1_backscatter_change", client, store)
    assert s1["status"] == "ok" and s1["ran"] == ["s1_backscatter_change"]

    downloads = len(client.requests)
    agree = _stored("s1_s2_agreement", client, store)
    assert agree["status"] == "ok" and agree["ran"] == ["s1_s2_agreement"]
    assert len(client.requests) == downloads            # maps came from disk
    assert agree["metrics"]["iou"] > 0.9
    assert agree["artifacts"] == [str(_study(tmp_path) / "burned_area_s2_vs_s1.png")]
    assert agree["study_dir"] == str(_study(tmp_path))


def test_a_saved_result_is_reused_unless_refresh(tmp_path):
    client, store = FakeSentinelHub(), tmp_path
    first = _stored("optical_conditions", client, store)
    n = len(client.requests)
    again = _stored("optical_conditions", client, store)
    assert again["reused"] and again["ran"] == [] and len(client.requests) == n
    assert again["summary"] == first["summary"]
    fresh = _stored("optical_conditions", client, store, refresh=True)
    assert not fresh["reused"] and len(client.requests) == n + 1


def test_a_failed_dependency_stops_the_chain_and_is_retried_later(tmp_path):
    store = tmp_path
    failed = _stored("s1_s2_agreement", FakeSentinelHub(fail=True), store)
    assert failed["status"] == "failed" and failed["ran"] == ["optical_conditions"]
    assert "optical_conditions" in failed["summary"]
    ok = _stored("s1_s2_agreement", FakeSentinelHub(), store)      # nothing ok was saved
    assert ok["status"] == "ok" and ok["ran"][0] == "optical_conditions"


def test_study_key_is_stable_and_separates_studies():
    key = an.AnalysisStore.study_key
    assert key(BBOX, START, END, 14) == key(list(BBOX), START + "T00:00", END, 14)
    assert key(BBOX, START, END, 14) != key(BBOX, START, END, 7)


# ------------------------------------------------------------------- the servers


@pytest.fixture
def analysis_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    from agenticeoecosystem.servers.cdse_analysis import server
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "_client", FakeSentinelHub())
    return server


def test_analysis_tools_run_and_report_the_study_dir(analysis_server, tmp_path):
    s = analysis_server
    out = asyncio.run(s.s2_dnbr_burned_area.fn(bbox=BBOX, start_date=START, end_date=END))
    assert out["status"] == "ok" and out["analysis"] == "s2_dnbr_burned_area"
    assert out["study_dir"].startswith(str(tmp_path))
    agree = asyncio.run(s.s1_s2_agreement.fn(bbox=BBOX, start_date=START, end_date=END))
    assert agree["ran"] == ["s1_backscatter_change", "s1_s2_agreement"]   # s2 was saved
    assert set(s.list_analyses.fn()) == set(an.ANALYSES)


@pytest.mark.parametrize("bbox,start,end,pad", [
    ([1, 2, 3], START, END, 14),
    ([10, 40, 9, 41], START, END, 14),
    (BBOX, "16/01/2026", END, 14),
    (BBOX, END, START, 14),
    (BBOX, START, END, 365),
])
def test_analysis_tools_reject_bad_inputs(analysis_server, bbox, start, end, pad):
    out = asyncio.run(analysis_server.optical_conditions.fn(bbox=bbox, start_date=start,
                                                            end_date=end, pad_days=pad))
    assert out["status"] == "failed" and out["error"]


def test_analysis_tools_need_cdse_credentials(analysis_server, monkeypatch):
    monkeypatch.setattr(analysis_server, "_client", None)
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    out = asyncio.run(analysis_server.optical_conditions.fn(bbox=BBOX, start_date=START, end_date=END))
    assert out["status"] == "failed" and "CDSE_CLIENT_ID" in out["error"]


@pytest.fixture
def research_server(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    from agenticeoecosystem.servers.eo_research import server
    d._LIST_CACHE.clear()
    monkeypatch.setattr(server, "_catalogs", _catalogs())
    monkeypatch.setattr(d, "_DEFAULT_CLIENT", FakeEmbedder())
    return server


def test_search_catalogues_matches_the_lab_tool(research_server):
    result = asyncio.run(research_server.search_catalogues.fn(
        query="radar images with phase", bbox=DISCOVERY_BBOX,
        start_date="2025-01-01", end_date="2025-03-31"))
    assert result["ranking"] == "embedding"
    assert any(c["id"] == "sentinel-1-slc" for c in result["collections"])
    assert all("availability" in c for c in result["collections"])


def test_search_catalogues_rejects_bad_inputs(research_server):
    s = research_server
    assert "error" in asyncio.run(s.search_catalogues.fn(query=" "))
    assert "error" in asyncio.run(s.search_catalogues.fn(query="radar", providers=["nasa"]))
    assert "error" in asyncio.run(s.search_catalogues.fn(query="radar", bbox=[0, 10, 1, 5]))


def test_geostationary_geometry_ranks_imagers_over_the_area(research_server):
    athens = research_server.geostationary_viewing_geometry.fn(bbox=[23.6, 37.9, 23.8, 38.1])
    best = athens["imagers"][0]
    assert best["verdict"] == "ok" and best["sub_satellite_lon"] in (0.0, 45.5)
    chile = research_server.geostationary_viewing_geometry.fn(bbox=[-72.7, -37.4, -72.6, -37.3])
    by_sensor = {r["sensor"]: r for r in chile["imagers"]}
    assert chile["imagers"][0]["sensor"] == "GOES-East ABI"
    assert by_sensor["MTG FCI"]["verdict"] in ("degraded", "unusable")
    assert by_sensor["GOES-East ABI"]["fetch_via"] == "no sub-agent in this system"
    assert "error" in research_server.geostationary_viewing_geometry.fn(bbox=[0, 1])


def test_rank_papers_tool_is_the_lab_function(research_server, monkeypatch):
    async def fake(ids: list[str]) -> dict:
        return {"ranked": [{"arxiv_id": i} for i in ids], "source": "semantic_scholar"}
    monkeypatch.setattr(research_server, "_rank_papers", fake)
    out = asyncio.run(research_server.rank_papers_by_citations.fn(arxiv_ids=["2106.09685"]))
    assert out["ranked"] == [{"arxiv_id": "2106.09685"}]


# ------------------------------------------------------------- shared geometry helpers


def test_bbox_helpers_handle_the_antimeridian():
    from agenticeoecosystem.agents.virtual_lab.sensor_catalog import bbox_centre, check_bbox
    fiji = [177.0, -19.0, -179.0, -16.0]                      # crosses the antimeridian
    assert check_bbox(fiji) is None and check_bbox(fiji, allow_antimeridian=False)
    lat, lon = bbox_centre(fiji)
    assert lat == -17.5 and lon == 179.0
    assert check_bbox([0, 10, 1, 5]) and check_bbox([1, 2, 3]) and check_bbox(None)
    assert check_bbox([5, 0, 5, 1])                          # zero width


def test_geometry_verdict_thresholds():
    from agenticeoecosystem.agents.virtual_lab.sensor_catalog import geometry_verdict
    assert [geometry_verdict(v) for v in (50, 70, 75, 80, 85)] == [
        "ok", "ok", "degraded", "degraded", "unusable"]


def test_servers_disagree_on_the_antimeridian_on_purpose(analysis_server, research_server):
    fiji = [177.0, -19.0, -179.0, -16.0]
    out = asyncio.run(analysis_server.optical_conditions.fn(bbox=fiji, start_date=START, end_date=END))
    assert out["status"] == "failed" and "antimeridian" in out["error"]
    geo = research_server.geostationary_viewing_geometry.fn(bbox=fiji)
    assert geo["centre"] == [-17.5, 179.0]
