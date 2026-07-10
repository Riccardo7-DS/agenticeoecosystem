from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from agenticeoecosystem.definitions import ROOT_DIR
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from mcp import StdioServerParameters
from typing import Optional

import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)



def _get_graphrag_config() -> dict:
    """Build the remote graphrag MCP server config (hosted via phi-gateway).

    Forwards the static Basic Auth header produced by graphrag/gen_auth_header.py;
    Basic Auth does not expire, so no token refresh is needed.
    """
    load_dotenv(ROOT_DIR / ".env")  # Ensure .env is loaded
    url = os.getenv("GRAPHRAG_MCP_URL")
    auth = os.getenv("GRAPHRAG_AUTH_HEADER")  # full value, e.g. "Basic <base64>"
    if not url or not auth:
        raise RuntimeError("Set GRAPHRAG_MCP_URL and GRAPHRAG_AUTH_HEADER.")
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": {"Authorization": auth},
    }

graphrag_config = _get_graphrag_config()
toolset = []
toolset = [
    McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=graphrag_config["url"],
            headers=graphrag_config["headers"],
        )
    )
]
_RAG_SYSTEM_PROMPT = (
    "You are a retrieval assistant for the ESA Earth Observation Science Strategy 2040 "
    "(EOPSS) GraphRAG knowledge base. You have three tools: "
    "search_eopss for natural-language questions (semantic chunk search + knowledge "
    "graph context), query_ontology for structured read-only Cypher lookups, and "
    "list_ontology_nodes to enumerate ontology nodes by type. "
    "Pick the appropriate tool, then return the raw retrieved context to the caller."
)

def _fetch_agent(toolset) -> Optional[Agent]:
    if toolset:
        return Agent(
            model=llm,
            name='graph_rag_agent',
            description='A helpful assistant for answering user questions related to ESA Earth Observation Science Strategy.',
            instruction=_RAG_SYSTEM_PROMPT,
            tools=toolset,
            mode='single_turn'
        )
    else:
        logger.warning("Graph RAG agent not initialized due to missing environment variables.")
        return None


graph_rag_agent = _fetch_agent(toolset)