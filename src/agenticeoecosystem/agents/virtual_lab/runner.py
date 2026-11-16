"""
runner.py — Entry points for running the Virtual Lab.

`get_ans_tool_path` returns the same JSON contract as
agents.multi_agent.get_ans_tool_path (final_answer, tools_called,
agents_called, tokens), plus the full `lab_record`, so pipeline_runner can
score pipeline="virtual_lab" against pipeline="multi" on the same tasks.
Everything is built per run: lab agents are cheap, and a fresh coordinator
avoids sharing a ledger between concurrent runs.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .config import VirtualLabConfig, build_eve_llm
from .coordinator import VirtualLabCoordinator
from .ledger import LabGuardPlugin, LabLedger
from .literature import ArxivSearch, ScholarSearch, rank_papers_by_citations
from .participants import build_participants, load_existing_agents
from .schemas import LabRecord

logger = logging.getLogger(__name__)

APP_NAME = "virtual_lab"  # matches the package, as ADK expects
USER_ID = "user_1"

_patches_applied = False


def _apply_runtime_patches() -> None:
    """Same LiteLLM/MCP patches multi_agent.py applies; needed for real EVE runs."""
    global _patches_applied
    if _patches_applied:
        return
    from agenticeoecosystem.agents.subagents import eve_patch, mcp_image_patch, strict_tool_patch
    from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard, arxiv_rate_limit_retry
    eve_patch.apply()
    mcp_image_patch.apply()
    strict_tool_patch.apply()
    orphaned_tool_call_guard.apply()
    arxiv_rate_limit_retry.apply()
    _patches_applied = True


def _default_artifact_dir() -> Path:
    from agenticeoecosystem.utils.output_paths import artifact_root
    return artifact_root() / "virtual_lab"


def build_lab(config: VirtualLabConfig | None = None, *, llm=None, existing_agents=None,
              extra_tools=None, artifact_dir: Path | None = None,
              literature_search=None, citation_ranker=rank_papers_by_citations,
              catalogs=None, analysis_client=None, embedder=None,
              ) -> tuple[Runner, VirtualLabCoordinator, LabLedger]:
    """Wire config, participants, coordinator, guard plugin and runner for one run.
    `llm`, `existing_agents`, `extra_tools`, `literature_search` and `citation_ranker`
    are injection points for tests. The literature check searches arxiv_rag_agent's
    MCP server by default and is skipped if that agent isn't loaded; with
    literature_backend='scholar' it searches scholar-search-mcp instead. `catalogs`
    (inventory.Catalogs) feeds the data inventory and the specialists' direct metadata tools;
    with real sub-agents it defaults to the public CDSE, EUMETSAT, NASA Earthdata and CDS APIs,
    with injected test agents to None, which skips both and keeps tests offline. `embedder`
    ranks the data discovery: by default the EMBEDDING_ENDPOINT client, or, with injected test
    agents, word overlap only (discovery.NO_EMBEDDINGS), so tests never depend on a service."""
    config = config or VirtualLabConfig.from_env()
    if llm is None or existing_agents is None:
        # Any real OpenAI-compatible endpoint or real sub-agent needs these.
        _apply_runtime_patches()
    if llm is None:
        llm = build_eve_llm(config)
    real_agents = existing_agents is None
    if existing_agents is None:
        existing_agents = load_existing_agents()
        if catalogs is None:
            from .inventory import Catalogs
            catalogs = Catalogs.public()
        if analysis_client is None and config.mode == "execute":
            from .analyses import SentinelHubClient
            analysis_client = SentinelHubClient.from_env()
    if embedder is None:
        from .discovery import NO_EMBEDDINGS, default_embedder
        embedder = default_embedder() if real_agents else NO_EMBEDDINGS
    ledger = LabLedger()
    participants = build_participants(config, llm, existing_agents, extra_tools, catalogs, embedder)
    if literature_search is None and config.literature_check:
        if config.literature_backend == "scholar":
            literature_search = ScholarSearch()
            if citation_ranker is rank_papers_by_citations:  # its results already carry counts
                citation_ranker = literature_search.rank
        else:
            literature_search = ArxivSearch.from_agents(existing_agents, config.literature_categories)
    coordinator = VirtualLabCoordinator(lab_config=config, participants=participants, ledger=ledger,
                                        literature_search=literature_search,
                                        citation_ranker=citation_ranker, catalogs=catalogs,
                                        analysis_client=analysis_client, embedder=embedder)
    runner = Runner(
        agent=coordinator,
        app_name=APP_NAME,
        session_service=InMemorySessionService(),
        artifact_service=FileArtifactService(artifact_dir or _default_artifact_dir()),
        plugins=[LabGuardPlugin(config, ledger, participants.names)],
    )
    return runner, coordinator, ledger


async def stream_virtual_lab(
    query: str, config: VirtualLabConfig | None = None, **build_kwargs: Any,
) -> AsyncGenerator[tuple[Event, VirtualLabCoordinator, LabLedger], None]:
    """Run one lab meeting, yielding (event, coordinator, ledger) for every ADK event, so a UI
    can render the ledger (proposals, critiques, tool calls) while the meeting runs."""
    from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard
    orphaned_tool_call_guard.reset()

    runner, coordinator, ledger = build_lab(config, **build_kwargs)
    session_id = f"lab_{uuid.uuid4().hex}"
    await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    content = types.Content(role="user", parts=[types.Part(text=query)])
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
            yield event, coordinator, ledger
    finally:  # a lab-owned MCP server (ScholarSearch) would otherwise outlive the run
        if (close := getattr(coordinator._search, "close", None)) is not None:
            await close()


async def run_virtual_lab(query: str, config: VirtualLabConfig | None = None, **build_kwargs
                          ) -> tuple[str, LabRecord | None, LabLedger, list[str]]:
    """Run one lab meeting. Returns (final_answer, record, ledger, agents_called)."""
    final_answer = "Virtual Lab did not produce a final response."
    agents_called: list[str] = []
    coordinator = ledger = None
    async for event, coordinator, ledger in stream_virtual_lab(query, config, **build_kwargs):
        if event.author and event.author != "user" and (not agents_called or agents_called[-1] != event.author):
            agents_called.append(event.author)
        if event.author == coordinator.name and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
            if text:
                final_answer = text
    return final_answer, coordinator.record if coordinator else None, ledger, agents_called


async def get_ans_tool_path(query: str) -> str:
    """pipeline_runner entry point; mirrors agents.multi_agent.get_ans_tool_path."""
    final_answer, record, ledger, agents_called = await run_virtual_lab(query)
    tools_called = []
    for call in ledger.tool_calls:
        if call["tool"].endswith("agent"):  # sub-agent invocations, not MCP tools (as in multi)
            continue
        entry = {"function_name": call["tool"], "arguments": call.get("args", {}),
                 "tool_result": call.get("result_preview")}
        if call["status"] in ("error", "blocked"):
            entry["error"] = call.get("reason") or call.get("result_preview")
        tools_called.append(entry)
    usage = ledger.usage()
    return json.dumps({
        "final_answer": final_answer,
        "ended_in_error": bool(tools_called) and "error" in tools_called[-1],
        "tools_called": tools_called,
        "agents_called": agents_called,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "lab_usage": usage,
        "lab_record": record.model_dump(mode="json") if record else None,
    }, default=str)
