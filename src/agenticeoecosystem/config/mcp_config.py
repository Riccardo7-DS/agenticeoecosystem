"""MCP (Model Context Protocol) servers configuration."""

import logging
import os

from dotenv import load_dotenv

from agenticeoecosystem.utils.output_paths import server_output_dir
from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv(dotenv_path=ROOT_DIR / ".env")

logger = logging.getLogger(__name__)

# gis-mcp is a third-party package that writes wherever a caller-supplied path
# resolves relative to its own process CWD, with no ROOT_DIR-aware rooting of
# its own. Pinning cwd here — the one lever we have over a subprocess we don't
# control the source of — makes every relative path gis-mcp resolves land
# under tool_output/gis_analysis/ instead of wherever this process was
# launched from. Mirrors the same fix for the "multi" pipeline in
# agents/subagents/sub_agent_gis.py.
_GIS_OUTPUT_DIR = server_output_dir("gis_analysis")
_GIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _try_register(servers: dict, key: str, build_fn) -> None:
    try:
        servers[key] = build_fn()
    except Exception as exc:
        logger.warning("%s MCP disabled: %s", key, exc)


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


def _check_cdse_env() -> None:
    """Raise RuntimeError if CDSE (Copernicus Data Space / Sentinel Hub)
    credentials are missing."""
    missing = [
        v for v in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET")
        if not os.getenv(v)
    ]
    if missing:
        raise RuntimeError(f"CDSE env vars not set: {', '.join(missing)} (check .env)")


def _check_gee_mcp_env() -> None:
    """Raise RuntimeError if GEE creds are missing."""
    cred_path = ROOT_DIR / "src/agenticeoecosystem/servers/gee/gee-key.json"
    if not cred_path.exists():
        raise RuntimeError("GEE credentials file not found.")


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
                "--storage-path", str(server_output_dir("arxiv"))
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

    # EVE retrieval is outside the benchmark toolset. Strategy questions use
    # eoss_graphrag, matching the multi-agent framework.

    _try_register(servers, "gis-mcp", lambda: {
        "transport": "stdio",
        "command": "uv",
        "args": ["tool", "run", "--python", "3.12", "gis-mcp"],
        "cwd": str(_GIS_OUTPUT_DIR),
    })

    _try_register(servers, "moocs", lambda: {
        # Local, file-based Qdrant store — no credentials required.
        "transport": "stdio",
        "command": "python",
        "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/moocs/server.py"],
    })

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

    # CDSE (Copernicus Data Space / Sentinel Hub) — also exposes the
    # Nominatim-backed geocode_place/resolve_named_region tools used
    # elsewhere as the preferred place-name resolver. The single-agent
    # pipeline had no route to this server at all before this entry (it
    # only reaches gee-mcp's Google-Maps-based get_lat_lon for imagery
    # requests), unlike the "multi" pipeline where sub_agent_cdse.py wires
    # cdse_agent directly and bypasses this registry.
    _try_register(servers, "cdse", lambda: (
        _check_cdse_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/cdse_geocode/server.py"],
        }
    ))

    _try_register(servers, "climate", lambda: (
        _check_climate_env() or {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/climate/server.py"],
        }
    ))

    # The Virtual Lab's tools as MCP servers. Opt-in (ENABLE_LAB_MCP_SERVERS=1): this
    # registry is the single-agent benchmark's toolset, and new tools would change it.
    if os.getenv("ENABLE_LAB_MCP_SERVERS", "").lower() in ("1", "true", "yes"):
        # Federated data discovery, citation ranking, geostationary geometry: public APIs.
        _try_register(servers, "eo_research", lambda: {
            "transport": "stdio",
            "command": "python",
            "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/eo_research/server.py"],
        })
        # Burned-area analyses on Sentinel Hub (CDSE processing quota).
        _try_register(servers, "cdse_analysis", lambda: (
            _check_cdse_env() or {
                "transport": "stdio",
                "command": "python",
                "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/cdse_analysis/server.py"],
            }
        ))

    logger.info("Final MCP servers: %s", list(servers.keys()))
    return servers
