from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from agenticeoecosystem.definitions import ROOT_DIR
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional
from pathlib import Path
import shutil
import os

import logging
logger = logging.getLogger(__name__)



_ARXIV_SYSTEM_PROMPT = (
    "You are an assistant for querying and retrieving scientific papers from an ArXiv MCP server. "
    "Use the available tools to search for papers, retrieve abstracts, download full texts, and perform semantic searches. "
    "When a user asks a question, determine which tool(s) to use and extract the necessary parameters from the query. "
    "For example, if a user asks 'Find recent papers on climate change and get their abstracts', you might use "
    "'search_papers' with the keyword 'climate change' and a recent date filter, and proper category, "
    "then use 'get_abstract' for each relevant paper found.\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Prefer 'get_abstract' over 'download_paper' unless the user explicitly requests the full text. "
    "  Abstracts contain enough information for most questions and avoid context overflows.\n"
    "- NEVER call 'download_paper' for more than one paper in a single task — full PDFs are very large "
    "  and will exceed the model's context window.\n"
    "- NEVER call 'get_abstract' for more than 5 papers in a single task; abstracts can still overflow context when batched too aggressively.\n"
    "- If the user only asks to find/fetch relevant papers, use discovery tools (e.g., search/list) and do not call 'get_abstract' or 'download_paper' unless explicitly requested.\n"
    "- If asked to summarise or analyse multiple papers, use 'get_abstract' for all of them.\n"
    "\n"
    "BUILDING search_papers QUERIES (this tool's own category list is CS/ML-biased and "
    "does not mention Earth Observation categories — do not copy it blindly):\n"
    "- 'query', 'categories', and any date filter are combined with AND — every one of "
    "them must match, and 'categories' itself is an OR across the list you give. Two "
    "quoted phrases ANDed together (e.g. '\"sea-level rise\" AND \"satellite altimetry\"') "
    "requires BOTH exact phrases to appear, which is often too strict and returns zero "
    "results even when relevant papers exist. Prefer ORing related phrasings instead, "
    "e.g. '\"sea level rise\" OR \"sea-level rise\"' or '\"satellite altimetry\" OR "
    "\"radar altimetry\" OR \"ocean altimetry\"'.\n"
    "- For Earth Observation / remote sensing / climate topics, the relevant arXiv "
    "categories are usually physics.ao-ph (Atmospheric and Oceanic Physics), "
    "physics.geo-ph (Geophysics), and sometimes eess.IV (Image/Video Processing) or "
    "eess.SP (Signal Processing) if the paper is about a processing technique rather "
    "than the science itself. Do NOT use eess.AS (Audio/Speech) or physics.data-an for "
    "EO topics — those are irrelevant despite appearing in the tool's example list.\n"
    "- If you are not confident which category fits, omit 'categories' entirely rather "
    "than guessing — an irrelevant category filter is more likely to zero out real "
    "results than to improve relevance.\n"
    "- If a search returns total_results: 0, retry ONCE with a broadened query before "
    "reporting no results to the user: switch any AND between phrases to OR, and/or "
    "drop the categories filter. Only report 'no papers found' after that retry also "
    "returns zero."
)

_ARXIV_MCP_BIN = shutil.which("arxiv-mcp-server") or str(
    Path.home() / ".local" / "bin" / "arxiv-mcp-server"
)


def _build_server_params() -> StdioServerParameters:
    """
    If the arxiv-mcp-server binary is already installed/discoverable, launch it
    directly (fast path). Otherwise — first run, or the package isn't installed
    yet — fall back to `uv tool run`, which will install it on demand before
    starting the server.
    """
    if os.path.isfile(_ARXIV_MCP_BIN) and os.access(_ARXIV_MCP_BIN, os.X_OK):
        logger.info(f"Found existing arxiv-mcp-server binary at {_ARXIV_MCP_BIN}; loading directly.")
        return StdioServerParameters(
            command=_ARXIV_MCP_BIN,
            args=["--storage-path", f"{ROOT_DIR}/output/arxiv"],
        )

    logger.info("arxiv-mcp-server binary not found; installing/running via 'uv tool run'.")
    return StdioServerParameters(
        command="uv",
        args=[
            "tool",
            "run",
            "--python", "3.12",
            "arxiv-mcp-server",
            "--storage-path", f"{ROOT_DIR}/output/arxiv",
        ],
    )


toolset = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=_build_server_params(),
            timeout=300,
        )
    )
]


def _fetch_agent(toolset) -> Optional[Agent]:
    if toolset:
        return Agent(
            model=llm,
            name='arxiv_rag_agent',
            description='A helpful assistant for answering user questions related to ArXiv paper retrieval.',
            instruction=_ARXIV_SYSTEM_PROMPT,
            tools=toolset,
            mode='single_turn'
        )
    else:
        logger.warning("ArXiv RAG agent not initialized due to missing toolset.")
        return None


arxiv_rag_agent = _fetch_agent(toolset)