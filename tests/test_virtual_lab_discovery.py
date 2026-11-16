"""
Offline tests of the Virtual Lab's federated data discovery (agents/virtual_lab/discovery.py):
listing every provider, ranking by meaning (a fake embedder) or word overlap, quotas, the CMR
keyword back-off, verification and the agents' search_catalogues tool.
"""

from __future__ import annotations

import asyncio

import pytest

from agenticeoecosystem.agents.virtual_lab import discovery as d
from agenticeoecosystem.agents.virtual_lab.inventory import Catalogs


@pytest.fixture(autouse=True)
def _pure_embedding_ranking(monkeypatch):
    """Rank by the fake embedder alone: the lexical bonus (DISCOVERY_HYBRID_WEIGHT) is a tunable
    whose default may change; these tests check listing, ranking by meaning and verification."""
    monkeypatch.setenv("DISCOVERY_HYBRID_WEIGHT", "0")

BBOX = [23.72, 37.98, 23.76, 38.02]


class FakeEmbedder:
    """Deterministic 'meaning': one axis per topic word, so similarity is topical overlap."""
    AXES = ("radar", "phase", "fire", "rain", "snow", "gnss")

    def __init__(self):
        self.calls = 0

    async def embed(self, texts, query=False):
        self.calls += 1
        out = []
        for t in texts:
            v = [1.0 if a in t.lower() else 0.0 for a in self.AXES] + [0.1]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


class Cdse:
    @staticmethod
    async def list_stac_collections(bbox=None, start_date=None, end_date=None):
        return {"total": 3, "matching": 3, "collections": [
            {"id": "sentinel-1-slc", "title": "Sentinel-1 Single Look Complex",
             "description": "Synthetic aperture radar images with phase", "keywords": ["radar"]},
            {"id": "sentinel-1-slc-wv", "title": "Sentinel-1 Single Look Complex: WV",
             "description": "radar wave mode with phase", "keywords": []},
            {"id": "snow-cover", "title": "Snow cover extent", "description": "snow", "keywords": []}]}

    @staticmethod
    async def search_stac_items(collection, bbox, start_date, end_date, **_):
        days = {"2025-01-05": ["04:22"], "2025-01-11": ["16:30"], "2025-01-17": ["04:22"]}
        return {"collection": collection, "total": None, "listed": 3, "days": days}


class Eumetsat:
    @staticmethod
    async def list_collections(query=None):
        return {"total": 3, "collections": [
            {"id": "EO:EUM:DAT:0682", "title": "Active Fire Monitoring (netCDF) - MTG"},
            {"id": "EO:EUM:DAT:0801", "title": "Active Fire Monitoring (CAP) - MTG"},
            {"id": "EO:EUM:DAT:0001", "title": "Rain rate - MSG"}]}

    @staticmethod
    async def search_products(collection_id, bbox, start_date, end_date):
        return {"total": 0, "listed": 0, "days": {}}


class Nasa:
    queries: list = []

    @classmethod
    async def search_collections(cls, keyword, bbox=None, start_date=None, end_date=None, limit=20):
        cls.queries.append(keyword)
        if len(keyword.split()) > 2:          # CMR ANDs words: long phrases find nothing
            return {"total": 0, "collections": []}
        return {"total": 1, "collections": [
            {"short_name": "SENTINEL-1_BURSTS", "title": "Sentinel-1 SLC bursts",
             "abstract": "radar phase bursts for interferometry"}]}

    @staticmethod
    async def search_granules(short_name, bbox, start_date, end_date):
        return {"error": "Earthdata granule search failed: 503"}


class Cds:
    @staticmethod
    async def list_datasets(start_date=None, end_date=None):
        return {"total": 1, "collections": [
            {"id": "insitu-observations-gnss", "title": "GNSS tropospheric delay",
             "description": "ground gnss water vapour", "keywords": [], "start": "1996-01-01", "end": None}]}

    @staticmethod
    async def dataset_info(dataset_id, start_date=None, end_date=None):
        return {"dataset": dataset_id, "title": "GNSS", "start": "1996-01-01", "end": None,
                "update_frequency": "Monthly", "covers_period": True}


def _catalogs():
    Nasa.queries = []
    return Catalogs(cdse=Cdse, eumetsat=Eumetsat, earthdata=Nasa, cds=Cds)


def _run(needs, catalogs=None, embedder=None, **kw):
    d._LIST_CACHE.clear()
    return asyncio.run(d.discover_for(needs, catalogs or _catalogs(), BBOX, "2025-01-01", "2025-03-31",
                                      d.direct_call, embedder or FakeEmbedder(), **kw))


def test_word_pairs_skip_function_words_and_spread_over_the_phrase():
    pairs = d._word_pairs("synthetic aperture radar single look complex images with phase, for "
                          "interferometric ground deformation measurement")
    assert len(pairs) == d.CMR_BACKOFF_QUERIES
    assert pairs[0] == "synthetic aperture" and pairs[-1] == "deformation measurement"
    assert not any(w in p.split() for p in pairs for w in ("with", "for"))
    assert d._word_pairs("radar phase") == ["radar phase"]


def test_every_provider_is_listed_ranked_by_meaning_and_verified():
    found = _run(["radar images with phase for interferometry", "ground gnss delay"])
    assert found.method == "embedding"
    ids = [h.candidate.id for h in found.hits]
    assert ids[:1] == ["sentinel-1-slc"] or ids[0] in ("sentinel-1-slc", "SENTINEL-1_BURSTS")
    assert {"sentinel-1-slc", "SENTINEL-1_BURSTS", "insitu-observations-gnss"} <= set(ids)
    assert "snow-cover" not in ids
    # CMR: the long need found nothing, so its word pairs were searched.
    assert Nasa.queries[0] == "radar images with phase for interferometry" and len(Nasa.queries) > 2
    by_id = {h.candidate.id: d.availability(r) for h, _, r in found.checks}
    assert "on 3 days" in by_id["sentinel-1-slc"] and "median revisit 6 d" in by_id["sentinel-1-slc"]
    assert "FAILED" in by_id["SENTINEL-1_BURSTS"] and "NOT absent" in by_id["SENTINEL-1_BURSTS"]
    assert "covers the study period" in by_id["insitu-observations-gnss"]


def test_format_variants_count_once_and_irrelevant_providers_add_nothing():
    found = _run(["fire detection"])
    eum = [h.candidate.id for h in found.hits if h.candidate.provider == "eumetsat"]
    assert len(eum) == 1 and eum[0] in ("EO:EUM:DAT:0682", "EO:EUM:DAT:0801")   # netCDF/CAP merged
    assert "EO:EUM:DAT:0001" not in eum                                        # rain: no match
    fire = next(r for h, _, r in found.checks if h.candidate.provider == "eumetsat")
    assert "NONE found" in d.availability(fire)


def test_word_overlap_ranking_when_embeddings_are_unavailable():
    found = _run(["radar phase"], embedder=d.NO_EMBEDDINGS)
    assert found.method == "lexical"
    assert "sentinel-1-slc" in [h.candidate.id for h in found.hits]
    assert "snow-cover" not in [h.candidate.id for h in found.hits]
    assert "word overlap" in d.render(found)


def test_listing_failures_are_reported_not_hidden():
    class DownEumetsat(Eumetsat):
        @staticmethod
        async def list_collections(query=None):
            return {"error": "EUMETSAT collection listing failed: 503"}

    found = _run(["radar phase"], catalogs=Catalogs(cdse=Cdse, eumetsat=DownEumetsat))
    text = d.render(found)
    assert "EUMETSAT Data Store: listing FAILED" in text
    assert "Copernicus Data Space (STAC): 3 collections considered" in text


def test_providers_filter_and_no_bbox_skips_verification():
    found = _run(["radar phase"], providers=["cdse"])
    assert {h.candidate.provider for h in found.hits} == {"cdse"}
    d._LIST_CACHE.clear()
    found = asyncio.run(d.discover_for(["radar phase"], _catalogs(), None, None, None, d.direct_call,
                                       FakeEmbedder()))
    assert found.hits and all(r is None for _, _, r in found.checks)
    assert "not verified" in d.render(found)


def test_listings_are_cached_between_runs():
    calls = []

    class CountingCdse(Cdse):
        @staticmethod
        async def list_stac_collections(**kw):
            calls.append(kw)
            return await Cdse.list_stac_collections(**kw)

    d._LIST_CACHE.clear()
    cats = Catalogs(cdse=CountingCdse)
    for _ in range(2):
        asyncio.run(d.discover_for(["radar"], cats, BBOX, "2025-01-01", "2025-03-31", d.direct_call,
                                   FakeEmbedder()))
    assert len(calls) == 1


def test_needs_fall_back_to_objectives():
    class P:
        data_needs: list = []
        objectives = ["Map subsidence", " "]
        question = "q"
    assert d.needs_of(P()) == ["Map subsidence"]
    P.data_needs = ["radar phase"]
    assert d.needs_of(P()) == ["radar phase"]


def test_search_catalogues_tool_returns_the_shortlist_with_availability():
    from agenticeoecosystem.agents.virtual_lab.participants import DISCOVERY, catalog_tools

    d._LIST_CACHE.clear()
    tools = catalog_tools(_catalogs(), FakeEmbedder())
    [tool] = tools[DISCOVERY]
    result = asyncio.run(tool.func(query="radar images with phase", bbox=BBOX,
                                   start_date="2025-01-01", end_date="2025-03-31"))
    assert result["ranking"] == "embedding"
    first = result["collections"][0]
    assert first["provider"] in d.PROVIDER_LABELS.values() and "availability" in first
    assert any(c["id"] == "sentinel-1-slc" for c in result["collections"])
