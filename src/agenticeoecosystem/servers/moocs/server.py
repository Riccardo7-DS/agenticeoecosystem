import logging
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "MOOCs_evepipeline"
EMBED_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EVE_URL = "https://phi-gateway.tce.ples.dedyn.io/neohpc-eve-internal/18000/v1"

mcp = FastMCP("MOOC-RAG")

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_qdrant: QdrantClient | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker %s", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_KEY must be set in .env")
        _qdrant = QdrantClient(url=url, api_key=api_key)
    return _qdrant


# -- Query rewriting via EVE (commented out — uncomment to improve retrieval at the cost of added latency) --
# import base64
# from openai import OpenAI
#
# _QUERY_REWRITE_PROMPT = (
#     "You are a search query optimizer for an Earth Observation MOOC knowledge base. "
#     "Rewrite the user's question into a precise, self-contained search query that "
#     "maximizes semantic similarity retrieval. Disambiguate ambiguous terms, "
#     "expand relevant acronyms, and make the intent explicit. "
#     "Return ONLY the rewritten query with no explanation."
# )
#
# def _rewrite_query(question: str) -> str:
#     user = os.getenv("EVE_USER")
#     password = os.getenv("EVE_PASSWORD")
#     if not user or not password:
#         logger.warning("EVE_USER/EVE_PASSWORD not set — skipping query rewrite")
#         return question
#     credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
#     eve = OpenAI(
#         base_url=EVE_URL,
#         api_key="EMPTY",
#         default_headers={"Authorization": f"Basic {credentials}"},
#     )
#     response = eve.chat.completions.create(
#         model="EVE-Instruct",
#         messages=[
#             {"role": "system", "content": _QUERY_REWRITE_PROMPT},
#             {"role": "user", "content": question},
#         ],
#         temperature=0,
#     )
#     rewritten = response.choices[0].message.content.strip()
#     logger.info("Query rewritten: %r -> %r", question, rewritten)
#     return rewritten
# -- end query rewriting --


def _retrieve(query: str, top_k: int = 20) -> list[tuple[float, dict]]:
    vector = _get_embedder().encode(query).tolist()
    results = _get_qdrant().query_points(
        collection_name=COLLECTION, query=vector, limit=top_k
    )
    return [(r.score, r.payload) for r in results.points]


def _rerank(
    query: str, hits: list[tuple[float, dict]], top_k: int = 5
) -> list[tuple[float, dict]]:
    pairs = [(query, h[1].get("content", "")) for h in hits]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(scores.tolist(), [h[1] for h in hits]), reverse=True)
    return ranked[:top_k]


def _format(hits: list[tuple[float, dict]]) -> str:
    if not hits:
        return "No relevant MOOC content found."
    lines = []
    for i, (score, payload) in enumerate(hits, start=1):
        title = payload.get("title", "unknown")
        year = payload.get("year", "")
        lines.append(
            f"[{i}] {title}{f' ({year})' if year else ''} — rerank score: {score:.4f}"
        )
        lines.append(payload.get("content", ""))
        lines.append("---")
    return "\n".join(lines)


@mcp.tool()
def search_moocs(query: str, top_k: int = 5) -> str:
    """Search MOOC transcripts for content semantically relevant to the query.

    Returns the top matching transcript chunks with their source title, year, and content.
    Use this for questions about Earth Observation concepts covered in educational courses.
    """
    logger.info("search_moocs: query=%r top_k=%d", query, top_k)
    try:
        # query = _rewrite_query(query)  # uncomment to enable EVE query rewriting
        candidates = _retrieve(query, top_k=20)
        top = _rerank(query, candidates, top_k=top_k)
        return _format(top)
    except Exception as e:
        logger.error("search_moocs error: %s", e, exc_info=True)
        return f"Error during retrieval: {e}"


if __name__ == "__main__":
    # mcp.run()
    _retrieve(query="sea ice")
