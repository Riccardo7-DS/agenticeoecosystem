"""MCP (Model Context Protocol) servers configuration."""

import logging
import os

import requests
from dotenv import load_dotenv

from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv(dotenv_path=ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


def _try_register(servers: dict, key: str, build_fn) -> None:
    try:
        servers[key] = build_fn()
    except Exception as exc:
        logger.warning("%s MCP disabled: %s", key, exc)


def _check_moocs_env() -> None:
    """Raise RuntimeError if required MOOC RAG env vars are missing."""
    missing = [v for v in ("QDRANT_URL", "QDRANT_KEY") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"MOOC RAG env vars not set: {', '.join(missing)} (check .env)")


def _check_geos_ops_env() -> None:
    """Raise RuntimeError if EUMETSAT Data Store credentials are missing."""
    missing = [
        v for v in ("EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET")
        if not os.getenv(v)
    ]
    if missing:
        raise RuntimeError(f"geos_ops (EUMETSAT) credentials not set: {', '.join(missing)}")


def _check_eoss_graphrag_env() -> None:
    """Raise RuntimeError if EOSS GraphRAG connection details are missing."""
    missing = [v for v in ("GRAPHRAG_MCP_URL", "GRAPHRAG_AUTH_HEADER") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"eoss_graphrag env vars not set: {', '.join(missing)} (check .env)")


def _check_climate_env() -> None:
    """Raise RuntimeError if CDS API credentials are missing."""
    missing = [v for v in ("CDS_URL", "CDS_KEY") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Climate (CDS) env vars not set: {', '.join(missing)} (check .env)")


_EVE_RETRIEVAL_URL = (
    "https://bedrock-agentcore.eu-west-1.amazonaws.com/runtimes/"
    "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-1%3A686812424034%3A"
    "runtime%2Feve_retrieval-GgFENBBHN0/invocations?qualifier=DEFAULT"
)

def _check_gee_mcp_env() -> None:
    """Raise RuntimeError if GEE creds are missing."""
    cred_path = ROOT_DIR / "src/agenticeoecosystem/servers/gee/gee-key.json"
    if not cred_path.exists():
        raise RuntimeError("GEE credentials file not found.")


def _get_eve_retrieval_token() -> str:
    client_id = os.getenv("EVE_RETRIEVAL_CLIENT_ID")
    client_secret = os.getenv("EVE_RETRIEVAL_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "EVE retrieval credentials are missing. Set EVE_RETRIEVAL_CLIENT_ID and EVE_RETRIEVAL_CLIENT_SECRET."
        )

    logger.info("Requesting EVE retrieval access token")
    resp = requests.post(
        url=os.getenv("EVE_RETRIEVAL_TOKEN_URL"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )
    resp.raise_for_status()
    logger.info("EVE retrieval token acquired")
    return resp.json()["access_token"]


def get_mcp_servers() -> dict:
    logger.info("Building MCP server list")
    servers = {
        "eodashboard": {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"],
        },
        "arxiv-mcp-server": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "tool",
                "run",
                "--python", "3.12",
                "arxiv-mcp-server",
                "--storage-path", f"{ROOT_DIR}/src/agenticeoecosystem/servers/arxiv_papers"
            ]
        }
    }
    try:
        _check_gee_mcp_env()
        servers["gee-mcp"] = {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/gee/server.py"],
            "env": {},
        }
    except Exception as exc:
        print(f"Warning: GEE MCP disabled: {exc}")

    _try_register(servers, "eve_retrieval", lambda: {
        "transport": "streamable_http",
        "url": _EVE_RETRIEVAL_URL,
        "headers": {"Authorization": f"Bearer {_get_eve_retrieval_token()}"},
    })

    _try_register(servers, "gis-mcp", lambda: {
        "transport": "stdio",
        "command": "uv",
        "args": ["tool", "run", "--python", "3.12", "gis-mcp"],
    })

    _try_register(servers, "moocs", lambda: (
        _check_moocs_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/moocs/server.py"],
        }
    ))

    _try_register(servers, "eoss_graphrag", lambda: (
        _check_eoss_graphrag_env() or {
            "transport": "streamable_http",
            "url": os.getenv("GRAPHRAG_MCP_URL"),
            "headers": {"Authorization": os.getenv("GRAPHRAG_AUTH_HEADER")},
        }
    ))

    _try_register(servers, "geos_ops", lambda: (
        _check_geos_ops_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/eumetsat/server.py"],
        }
    ))

    _try_register(servers, "climate", lambda: (
        _check_climate_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/climate/server.py"],
        }
    ))

    logger.info("Final MCP servers: %s", list(servers.keys()))
    return servers
