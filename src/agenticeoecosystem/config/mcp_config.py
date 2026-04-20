"""MCP (Model Context Protocol) servers configuration."""

import os

import requests
from dotenv import load_dotenv

from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv(dotenv_path=ROOT_DIR / ".env")

_EVE_RETRIEVAL_URL = (
    "https://bedrock-agentcore.eu-west-1.amazonaws.com/runtimes/"
    "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-1%3A686812424034%3A"
    "runtime%2Feve_retrieval-AAvF6o9cAx/invocations?qualifier=DEFAULT"
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
        "https://eve-tools.auth.eu-west-1.amazoncognito.com/oauth2/token",
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

    print(f"[mcp_config] Final MCP servers: {list(servers.keys())}", flush=True)
    return servers


# Keep MCP_SERVERS for backwards compatibility — callers that don't need
# eve_retrieval can still import this without a network call.
MCP_SERVERS = {
    # "example": {
    #     "transport": "stdio",
    #     "command": "python",
    #     "args": [
    #         f"{ROOT_DIR}/src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py"
    #     ],
    # },
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
