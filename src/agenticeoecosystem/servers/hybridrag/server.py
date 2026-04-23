"""MCP server exposing hybridrag retrieval (Qdrant + Neo4j) as a tool."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

HYBRIDRAG_ROOT = Path(__file__).resolve().parents[5] / "hybridrag_pipeline"
load_dotenv(HYBRIDRAG_ROOT / ".env")
sys.path.insert(0, str(HYBRIDRAG_ROOT))

# MCP stdio transport owns stdout for JSON-RPC — nothing else may write there.
# hybridrag's setup_logging() (called at import time, with force=True) adds a
# StreamHandler(sys.stdout). By pointing sys.stdout at sys.stderr for the
# duration of the import, setup_logging() binds its handler to stderr instead,
# keeping stdout clean for MCP from that point on.
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    from query_pipeline import init_pipeline, retrieve  # noqa: E402
finally:
    sys.stdout = _real_stdout

mcp = FastMCP("hybridrag")

try:
    _qdrant, _driver, _embedder = init_pipeline()
    _init_error: str | None = None
except Exception as exc:
    _qdrant = _driver = _embedder = None
    _init_error = str(exc)


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
    if _init_error:
        return f"Retrieval unavailable (pipeline failed to initialise: {_init_error})"
    try:
        import logging
        logging.getLogger(__name__).info("[hybrid_rag_retrieve] question: %s", question)
        return retrieve(question, qdrant=_qdrant, driver=_driver, embedder=_embedder)
    except Exception as exc:
        return f"Retrieval error: {exc}"


if __name__ == "__main__":
    mcp.run()
