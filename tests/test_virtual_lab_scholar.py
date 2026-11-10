"""ScholarSearch (scholar-search-mcp backend of the literature check), offline with a fake toolset."""

import asyncio
import json

from agenticeoecosystem.agents.virtual_lab.config import VirtualLabConfig
from agenticeoecosystem.agents.virtual_lab.literature import ScholarSearch, plain_query, scholar_paper

S2_HIT = {"paperId": "a" * 40, "title": "Burn severity from Sentinel-2", "abstract": "dNBR ...",
          "year": 2021, "publicationDate": "2021-03-01", "citationCount": 120,
          "influentialCitationCount": 9, "externalIds": {"ArXiv": "2103.01234", "DOI": "10.1/x"},
          "source": "semantic_scholar"}
ARXIV_HIT = {"paperId": "2205.04321", "title": "SAR wildfire mapping", "abstract": "C-band ...",
             "year": 2022, "citationCount": None, "source": "arxiv"}
JOURNAL_ONLY = {"paperId": "b" * 40, "title": "RSE paper", "citationCount": 500,
                "externalIds": {"DOI": "10.1016/j.rse"}, "source": "semantic_scholar"}


def _ctx():
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.sessions import InMemorySessionService

    svc = InMemorySessionService()
    session = asyncio.run(svc.create_session(app_name="t", user_id="u"))
    return InvocationContext(session_service=svc, invocation_id="i", agent=LlmAgent(name="x"),
                             session=session)


class _Tool:
    name = "search_papers"

    def __init__(self, data):
        self.data, self.calls = data, []

    async def run_async(self, args, tool_context):
        self.calls.append(args)
        return {"content": [{"type": "text", "text": json.dumps({"total": 3, "data": self.data})}]}


class _Toolset:
    def __init__(self, tool):
        self.tool, self.closed = tool, False

    async def get_tools(self):
        return [self.tool]

    async def close(self):
        self.closed = True


def test_plain_query_strips_arxiv_syntax():
    assert plain_query('ti:"burn severity" AND (dNBR OR RBR)') == "burn severity dNBR RBR"


def test_scholar_paper_keeps_only_arxiv_ids():
    assert scholar_paper(S2_HIT)["id"] == "2103.01234"
    assert scholar_paper(ARXIV_HIT)["id"] == "2205.04321"
    assert scholar_paper(JOURNAL_ONLY) is None


def test_search_and_rank(monkeypatch):
    tool = _Tool([S2_HIT, JOURNAL_ONLY, ARXIV_HIT])
    search = ScholarSearch(_Toolset(tool))
    papers = asyncio.run(search(_ctx(), 'ti:"burn severity" AND dNBR', 5))
    assert [p["id"] for p in papers] == ["2103.01234", "2205.04321"]
    assert tool.calls[0]["query"] == "burn severity dNBR" and tool.calls[0]["limit"] == 5

    ranked = asyncio.run(search.rank(["2103.01234"]))  # count known: no network lookup
    assert ranked["ranked"][0]["citation_count"] == 120

    async def fake_lookup(ids):
        return {"ranked": [{"arxiv_id": i} for i in ids], "source": "openalex"}
    monkeypatch.setattr("agenticeoecosystem.agents.virtual_lab.literature.rank_papers_by_citations",
                        fake_lookup)
    assert asyncio.run(search.rank(["2103.01234", "2205.04321"]))["source"] == "openalex"


def test_backend_from_env(monkeypatch):
    monkeypatch.setenv("VIRTUAL_LAB_LITERATURE_BACKEND", "Scholar")
    assert VirtualLabConfig.from_env().literature_backend == "scholar"
    monkeypatch.delenv("VIRTUAL_LAB_LITERATURE_BACKEND")
    assert VirtualLabConfig.from_env().literature_backend == "arxiv"
