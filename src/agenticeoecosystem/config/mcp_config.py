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


def _get_graphrag_config() -> dict:
    """Build the remote graphrag MCP server config (hosted via phi-gateway).

    Forwards the static Basic Auth header produced by graphrag/gen_auth_header.py;
    Basic Auth does not expire, so no token refresh is needed.
    """
    url = os.getenv("GRAPHRAG_MCP_URL")
    auth = os.getenv("GRAPHRAG_AUTH_HEADER")  # full value, e.g. "Basic <base64>"
    if not url or not auth:
        raise RuntimeError("Set GRAPHRAG_MCP_URL and GRAPHRAG_AUTH_HEADER.")
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": {"Authorization": auth},
    }


def _check_moocs_env() -> None:
    """Raise RuntimeError if required MOOC RAG env vars are missing."""
    missing = [v for v in ("QDRANT_URL", "QDRANT_KEY") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"MOOC RAG env vars not set: {', '.join(missing)} (check .env)")


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

    _try_register(servers, "graphrag", _get_graphrag_config)

    _try_register(servers, "moocs", lambda: (
        _check_moocs_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/moocs/server.py"],
        }
    ))

    logger.info("Final MCP servers: %s", list(servers.keys()))
    return servers
