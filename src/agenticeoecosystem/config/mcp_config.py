"""MCP (Model Context Protocol) servers configuration."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv(dotenv_path=ROOT_DIR / ".env")

_HYBRIDRAG_ROOT = ROOT_DIR.parent / "hybridrag_pipeline"


def _check_hybridrag_env() -> None:
    """Raise RuntimeError if required hybridrag env vars are missing."""
    load_dotenv(_HYBRIDRAG_ROOT / ".env")
    missing = [
        v for v in ("QDRANT_URL", "QDRANT_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        if not os.getenv(v)
    ]
    if missing:
        raise RuntimeError(
            f"hybridrag env vars not set: {', '.join(missing)} (check hybridrag_pipeline/.env)"
        )

_EVE_RETRIEVAL_URL = (
    "https://bedrock-agentcore.eu-west-1.amazonaws.com/runtimes/"
    "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-1%3A686812424034%3A"
    "runtime%2Feve_retrieval-GgFENBBHN0/invocations?qualifier=DEFAULT"
)


def _get_eve_retrieval_token() -> str:
    client_id = os.getenv("EVE_RETRIEVAL_CLIENT_ID")
    client_secret = os.getenv("EVE_RETRIEVAL_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "EVE retrieval credentials are missing. Set EVE_RETRIEVAL_CLIENT_ID and EVE_RETRIEVAL_CLIENT_SECRET."
        )

    print("[mcp_config] Requesting EVE retrieval access token...", flush=True)
    resp = requests.post(
        url= "https://eu-west-1xu6esehdg.auth.eu-west-1.amazoncognito.com/oauth2/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )
    resp.raise_for_status()
    access_token = resp.json()["access_token"]
    print("[mcp_config] EVE retrieval token acquired.", flush=True)
    return access_token


def get_mcp_servers() -> dict:
    print("[mcp_config] Building MCP server list...", flush=True)
    servers = {
        "eodashboard": {
            "transport": "stdio",
            "command": "python",
            "args": [
                f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"
            ],
        },
        "gee-mcp": {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/gee/server.py"],
            "env": {},
        },
    }

    # Do not block startup if token retrieval is unavailable.
    try:
        token = _get_eve_retrieval_token()
        servers["eve_retrieval"] = {
            "transport": "streamable_http",
            "url": _EVE_RETRIEVAL_URL,
            "headers": {"Authorization": f"Bearer {token}"},
        }
    except Exception as exc:
        print(f"Warning: eve_retrieval MCP disabled: {exc}")

    # Do not block startup if Qdrant/Neo4j are unavailable.
    try:
        _check_hybridrag_env()
        servers["get_hybrid_rag_context "] = {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/hybridrag/server.py"],
        }
    except Exception as exc:
        print(f"Warning: hybridrag MCP disabled: {exc}")

    print(f"[mcp_config] Final MCP servers: {list(servers.keys())}", flush=True)
    return servers


# # Keep MCP_SERVERS for backwards compatibility — callers that don't need
# # eve_retrieval can still import this without a network call.
# MCP_SERVERS = {
#     # "example": {
#     #     "transport": "stdio",
#     #     "command": "python",
#     #     "args": [
#     #         f"{ROOT_DIR}/src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py"
#     #     ],
#     # },
#     "eodashboard": {
#         "transport": "stdio",
#         "command": "python",
#         "args": [
#             f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"
#         ],
#     },
#     "gee-mcp": {
#         "transport": "stdio",
#         "command": "python",
#         "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/gee/server.py"],
#         "env": {},
#     },
# }
