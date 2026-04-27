"""MCP server exposing hybridrag retrieval (Qdrant + Neo4j) as a tool."""

import contextlib
import logging
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agenticeoecosystem.definitions import HYBRIDRAG_ROOT

load_dotenv(HYBRIDRAG_ROOT / ".env")
sys.path.insert(0, str(HYBRIDRAG_ROOT))

# setup_logging() in query_pipeline fires at import time and binds a StreamHandler to
# sys.stdout — redirect so it lands on stderr, keeping the MCP JSON-RPC channel clean.
with contextlib.redirect_stdout(sys.stderr):
    from query_pipeline import init_pipeline, retrieve  # type: ignore[import-not-found]

mcp = FastMCP("hybridrag")
_qdrant, _driver, _embedder = init_pipeline()

logger = logging.getLogger(__name__)


@mcp.tool()
def hybrid_rag_retrieve(question: str) -> str:
    """Retrieve relevant document chunks and ESA knowledge graph context for a question.

    Returns formatted text with retrieved PDF chunks and graph relationships from the
    ESA EO Science Strategy knowledge base. Use this for questions about ESA science
    themes, science questions, strategic objectives, enabling tools, tipping points,
    treaties, and related Earth Observation topics.

    Args:
        question: The natural language question to retrieve context for.
    """
    logger.info("[hybrid_rag_retrieve] question: %s", question)
    return retrieve(question, qdrant=_qdrant, driver=_driver, embedder=_embedder)


if __name__ == "__main__":
    mcp.run()
