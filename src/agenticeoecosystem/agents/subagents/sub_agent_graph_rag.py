from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from agenticeoecosystem.definitions import ROOT_DIR
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from mcp import StdioServerParameters
from typing import Optional

import os
import httpx
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)


_MCP_REQUEST_TIMEOUT_SEC = 30.0
_MCP_SSE_READ_TIMEOUT_SEC = 300.0
_MCP_CONNECT_RETRIES = 3


def _create_retrying_mcp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Create MCP HTTP client with retries for transient gateway/connect errors."""
    kwargs: dict = {
        "follow_redirects": True,
        "transport": httpx.AsyncHTTPTransport(retries=_MCP_CONNECT_RETRIES),
    }

    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(
            _MCP_REQUEST_TIMEOUT_SEC,
            read=_MCP_SSE_READ_TIMEOUT_SEC,
        )
    else:
        kwargs["timeout"] = timeout

    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth

    return httpx.AsyncClient(**kwargs)



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
            timeout=_MCP_REQUEST_TIMEOUT_SEC,
            sse_read_timeout=_MCP_SSE_READ_TIMEOUT_SEC,
            httpx_client_factory=_create_retrying_mcp_http_client,
        )
    )
]
_RAG_SYSTEM_PROMPT = (
    "You are a retrieval assistant for the ESA Earth Observation Science Strategy "
    "knowledge graph. Choose the right tool for the request:\n"
    "- search_eoss: natural-language questions; returns relevant document chunks "
    "plus their knowledge-graph context.\n"
    "- query_ontology: structured, read-only Cypher lookups (MATCH ... RETURN) "
    "over the graph.\n"
    "- list_ontology_nodes: enumerate nodes of a given label (e.g. "
    "StrategicObjective, ScienceQuestion, KnowledgeAdvancementObjective), or "
    "list all node types with counts when no label is given.\n"
    "Call the appropriate tool, then return the raw context to the caller."
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