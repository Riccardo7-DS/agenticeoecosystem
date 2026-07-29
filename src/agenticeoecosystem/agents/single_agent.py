"""Single-agent baseline: one ADK Agent with every tool domain attached
directly (no routing/delegation layer), for comparison against the
multi-agent supervisor in multi_agent.py.

Unlike an earlier version of this file, toolsets are NOT imported from
agents/subagents/sub_agent_*.py. Importing anything from that package
triggers subagents/__init__.py's full pkgutil scan, which builds and prints
all 10 multi-agent sub-agents ("Loaded 10 sub-agents: [...]") regardless of
which single submodule you actually wanted — misleading noise when running
this single-agent baseline, since those Agent objects are irrelevant here.
So every MCP server toolset below is constructed directly, inline, from the
same command/args/timeout each sub_agent_*.py file uses (verified against
those files directly) — not re-implemented, just not routed through that
package. The only things still imported from agents/subagents/ are the 4
generic litellm/ADK monkeypatches (eve_patch, mcp_image_patch,
orphaned_tool_call_guard, arxiv_rate_limit_retry), since duplicating ~650
lines of process-wide bugfixes here would be worse than the log noise —
their one-time import-triggered print is suppressed below instead (see
_import_adk_patches_quietly), without touching that shared file.

Each toolset is built through _safe_build_toolset(), which logs a clear
per-domain success/failure line and degrades to an empty toolset (with a
warning) rather than letting one missing credential crash the whole agent —
so e.g. a missing GRAPHRAG_MCP_URL just means GraphRAG tools are unavailable,
not an ImportError for the entire file.

Output paths (GIS working dir, ArXiv paper storage) are rooted at
Path.cwd() — wherever this process is actually launched from — rather than
the repo's ROOT_DIR, so a benchmark run started from a different working
directory writes its outputs there, not into the repo checkout.

The system prompt below is a condensed synthesis of the 10 sub-agent
instructions plus the orchestrator's routing/output/scope rules, since a
single agent (unlike the supervisor) has no sub-agent to hand domain-specific
know-how off to and must carry all of it itself.

VLM handling: `llm` (EVE-Instruct) is not known to be multimodal, so this
agent does NOT get the raw, image-bytes-returning `load_images` tool (that
one is only safe for sub_agent_vlm.py's separate vision-capable model, see
vlm_init.py — untouched). Instead it gets the same server's text-only
`resolve_images` tool plus one new self-contained tool, `describe_images`,
which resolves + loads images and asks a vision-language model the question
INTERNALLY, returning only a plain-text answer.
"""

import contextlib
import io
import logging

# The 4 patches below live under agents/subagents/, whose package
# __init__.py eagerly imports every sub_agent_*.py the moment ANY submodule
# in that package is imported — logging "Loaded sub-agent 'X' from Y" for
# each one (plus a final print of the full roster) regardless of which
# submodule was actually requested. All 10 are irrelevant to this
# single-agent baseline and confusing to see in its log. Silence both the
# print AND the logger calls for this one-time import (logging.disable
# mutes every logger process-wide, but only for this narrow window —
# restored immediately after) without modifying the shared package file.
logging.disable(logging.CRITICAL)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from agenticeoecosystem.agents.subagents.eve_patch import apply as _apply_eve_patch
        from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as _apply_mcp_image_patch
        from agenticeoecosystem.agents.subagents import orphaned_tool_call_guard as _orphan_guard
        from agenticeoecosystem.agents.subagents import arxiv_rate_limit_retry as _arxiv_retry
finally:
    logging.disable(logging.NOTSET)

_apply_eve_patch()
_apply_mcp_image_patch()
_orphan_guard.apply()
_arxiv_retry.apply()

import asyncio
import base64
import json
import os
import shutil
import subprocess
import uuid
import warnings
from pathlib import Path
from typing import Callable

import litellm
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from mcp import StdioServerParameters
from dotenv import load_dotenv

from agenticeoecosystem.definitions import ROOT_DIR

# Reuse vlm_tools/server.py's own (private, unmodified) discovery/GeoTIFF
# helpers so describe_images loads images identically to load_images —
# without editing that shared file. This import is from agents/servers/,
# not agents/subagents/, so it doesn't trigger the noisy package init above.
from agenticeoecosystem.servers.vlm_tools.server import (
    _discover as _vlm_discover,
    _geotiff_to_png_bytes,
    _GEOTIFF_EXTENSIONS,
    _MIME_TYPES,
)

warnings.filterwarnings("ignore")
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths — relative to wherever this process is actually run from
# (Path.cwd()), not the repo's ROOT_DIR, so a run launched from a different
# working directory writes its outputs there.
# ---------------------------------------------------------------------------

_RUN_DIR = Path.cwd()
_GIS_OUTPUT_DIR = _RUN_DIR / "output" / "gis_analysis"
_ARXIV_STORAGE_DIR = _RUN_DIR / "output" / "arxiv"
_GIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_ARXIV_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return "<not set>"
    return value[:visible] + "*" * max(0, len(value) - visible)


def _basic_auth_header(username_env: str = "EVE_USERNAME", password_env: str = "EVE_PASSWORD") -> dict:
    username = os.getenv(username_env, "")
    password = os.getenv(password_env, "")
    if not username or not password:
        logger.warning(
            "Credentials not found: %s=%s, %s=%s",
            username_env, _mask_secret(username), password_env, _mask_secret(password),
        )
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


_MODEL_NAME = os.getenv("MULTI_AGENT_MODEL", "EVE-Instruct")
llm = LiteLlm(
    model=f"openai/{_MODEL_NAME}",
    base_url=os.getenv("EVE_ENDPOINT"),
    headers=_basic_auth_header(),
    api_key="EMPTY",
    temperature=0,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Toolset construction — each domain built directly (no subagents import),
# wrapped so a failure logs clearly and degrades to no tools for that domain
# instead of crashing the whole file.
# ---------------------------------------------------------------------------

def _safe_build_toolset(label: str, builder: Callable[[], list]) -> list:
    try:
        toolset = builder()
        logger.info("[single_agent] tool domain '%s' loaded OK", label)
        return toolset
    except Exception as e:
        logger.warning("[single_agent] tool domain '%s' FAILED to load — it will be "
                        "unavailable to this agent: %s", label, e)
        return []


def _ensure_uv_tool_installed(tool_name: str) -> None:
    """Make sure `tool_name` is installed as a `uv tool`, so the first real
    MCP handshake doesn't stall on a cold on-demand install. `uv tool
    install` is idempotent — safe to call every time this module loads."""
    try:
        subprocess.run(
            ["uv", "tool", "install", tool_name],
            check=True, timeout=300, capture_output=True, text=True,
        )
        logger.info("[single_agent] uv tool '%s' is installed", tool_name)
    except Exception as e:
        logger.warning(
            "[single_agent] could not confirm/install uv tool '%s' (%s) — "
            "will still try to run it, which may install it on demand and "
            "delay the first call to that tool.", tool_name, e,
        )


# --- CDSE (Copernicus Data Space Ecosystem: Sentinel/Landsat/DEM + geocoding) ---
_cdse_toolset = _safe_build_toolset("CDSE", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/cdse_geocode/server.py"],
        ),
        timeout=300,
    ))
])

# --- Climate (ERA5 reanalysis via Copernicus CDS) ---
_climate_toolset = _safe_build_toolset("Climate (ERA5/CDS)", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/climate/server.py"],
        ),
        # CDS queues large/hourly requests server-side; cdsapi's own client
        # polls for a long time (retry_max=500, sleep_max=120s) — match
        # sub_agent_climate.py's timeout so long ERA5 fetches aren't killed
        # by the MCP connection before cdsapi itself would give up.
        timeout=1800,
    ))
])

# --- EO Dashboard (narratives & STAC catalog) ---
_eodashboard_toolset = _safe_build_toolset("EO Dashboard", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"],
        ),
        timeout=300,
    ))
])

# --- EUMETSAT (MTG FCI / MSG SEVIRI geostationary products) ---
_eumetsat_toolset = _safe_build_toolset("EUMETSAT", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/eumetsat/server.py"],
        ),
        timeout=300,
    ))
])

# --- GEE (Google Earth Engine) ---
_gee_toolset = _safe_build_toolset("GEE", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/gee/server.py"],
        ),
        timeout=300,
    ))
])

# --- GIS analysis (external gis-mcp package: rasterio/geopandas/shapely/pyproj/pysal) ---
# Same allow-list as sub_agent_gis.py's _GIS_ALLOWED_TOOL_NAMES — excludes
# gis-mcp's own data-download tools (download_boundaries, download_worldcover,
# etc.), which duplicate what CDSE/GEE/EUMETSAT/Climate already cover here.
_GIS_TOOL_NAMES = {
    "buffer", "intersection", "union", "difference", "symmetric_difference",
    "convex_hull", "envelope", "minimum_rotated_rectangle", "centroid",
    "bounds", "coordinates", "geometry_type", "rotate", "scale", "translate",
    "triangulate", "voronoi", "unary_union", "length", "area",
    "is_valid", "make_valid", "simplify", "snap", "nearest_point",
    "normalize", "geometry_to_geojson", "geojson_to_geometry",
    "transform_coordinates", "project_geometry",
    "get_crs_info", "get_available_crs", "get_utm_zone", "get_utm_crs", "get_geocentric_crs",
    "get_geod_info", "calculate_geodetic_distance", "calculate_geodetic_point", "calculate_geodetic_area",
    "read_file_gpd", "write_file_gpd", "append_gpd", "merge_gpd",
    "overlay_gpd", "dissolve_gpd", "explode_gpd", "clip_vector",
    "sjoin_gpd", "sjoin_nearest_gpd", "point_in_polygon",
    "metadata_raster", "get_raster_crs", "extract_band", "raster_band_statistics",
    "raster_histogram", "clip_raster_with_shapefile", "resample_raster",
    "reproject_raster", "tile_raster", "compute_ndvi", "raster_algebra",
    "concat_bands", "weighted_band_sum", "zonal_statistics",
    "reclassify_raster", "focal_statistics", "hillshade", "write_raster",
    "morans_i", "gearys_c", "gamma_statistic", "getis_ord_g",
    "moran_local", "getis_ord_g_local", "join_counts_local", "adbscan",
    "weights_from_shapefile", "distance_band_weights", "knn_weights",
    "build_transform_and_save_weights", "ols_with_spatial_diagnostics_safe",
    "build_and_transform_weights", "spatial_markov", "dynamic_lisa", "gm_lag",
    "create_map", "create_web_map",
    "download_boundaries", "download_climate_data",
    "get_species_info", "download_species_occurrences",
    "download_street_network", "calculate_shortest_path",
    "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
}
_GIS_DOWNLOAD_TOOL_NAMES = {
    "download_boundaries", "download_climate_data",
    "get_species_info", "download_species_occurrences",
    "download_street_network", "calculate_shortest_path",
    "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
}
_GIS_ALLOWED_TOOL_NAMES = sorted(_GIS_TOOL_NAMES - _GIS_DOWNLOAD_TOOL_NAMES)

_ensure_uv_tool_installed("gis-mcp")

_gis_toolset = _safe_build_toolset("GIS", lambda: [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uv",
                args=["tool", "run", "--python", "3.12", "gis-mcp"],
                cwd=str(_GIS_OUTPUT_DIR),
            ),
            timeout=300,
        ),
        tool_filter=_GIS_ALLOWED_TOOL_NAMES,
    )
])


def clip_raster_with_bbox(
    raster_path: str,
    bbox: list[float],
    destination: str,
    bbox_crs: str = "EPSG:4326",
) -> dict:
    """Clip a raster to a bounding box, without requiring a shapefile.

    Args:
        raster_path: Local path to the source raster.
        bbox: [west, south, east, north].
        destination: Local path to write the clipped raster to.
        bbox_crs: CRS of the bbox coordinates (default WGS-84 lon/lat).

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    src_path = os.path.expanduser(raster_path.replace("`", ""))
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Raster not found at '{src_path}'.")

    west, south, east, north = bbox

    with rasterio.open(src_path) as src:
        if src.crs and str(src.crs) != bbox_crs:
            west, south, east, north = transform_bounds(bbox_crs, src.crs, west, south, east, north)
        window = from_bounds(west, south, east, north, transform=src.transform)
        out_transform = src.window_transform(window)
        out_image = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    dst_path = Path(os.path.expanduser(destination.replace("`", "")))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(dst_path), "w", **out_meta) as dst:
        dst.write(out_image)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"Raster clipped to bbox and saved to '{dst_path}'.",
    }


def compute_evi(
    source: str,
    red_band_index: int,
    nir_band_index: int,
    blue_band_index: int,
    destination: str,
    scale_factor: float = 0.0001,
) -> dict:
    """Compute EVI (Enhanced Vegetation Index) and save to GeoTIFF.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B4, B2).
        red_band_index: Index of the red band (1-based).
        nir_band_index: Index of the near-infrared band (1-based).
        blue_band_index: Index of the blue band (1-based).
        destination: Output EVI raster path.
        scale_factor: Multiplier applied to raw band values to convert them
            to reflectance (0-1) before applying the formula. Sentinel-2 SR
            products store DN 0-10000, so the default 0.0001 is correct for
            them; pass 1.0 if the source raster is already reflectance.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        red = src.read(red_band_index).astype("float32") * scale_factor
        nir = src.read(nir_band_index).astype("float32") * scale_factor
        blue = src.read(blue_band_index).astype("float32") * scale_factor
        denom = nir + 6 * red - 7.5 * blue + 1
        evi = 2.5 * (nir - red) / np.where(denom == 0, np.nan, denom)
        evi = np.nan_to_num(evi, nan=0.0).astype("float32")

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(evi, 1)

    return {"status": "success", "destination": str(dst_path),
            "message": f"EVI calculated and saved to '{dst_path}'."}


def compute_ndwi(source: str, green_band_index: int, nir_band_index: int, destination: str) -> dict:
    """Compute NDWI (McFeeters Normalized Difference Water Index) and save to GeoTIFF.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B3, B8).
        green_band_index: Index of the green band (1-based).
        nir_band_index: Index of the near-infrared band (1-based).
        destination: Output NDWI raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        green = src.read(green_band_index).astype("float32")
        nir = src.read(nir_band_index).astype("float32")
        ndwi = (green - nir) / (green + nir + 1e-6)
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(ndwi, 1)

    return {"status": "success", "destination": str(dst_path),
            "message": f"NDWI calculated and saved to '{dst_path}'."}


def compute_ndmi(source: str, nir_band_index: int, swir1_band_index: int, destination: str) -> dict:
    """Compute NDMI (Normalized Difference Moisture Index) and save to GeoTIFF.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B11).
        nir_band_index: Index of the near-infrared band (1-based).
        swir1_band_index: Index of the SWIR1 band (1-based, ~1.6um; Sentinel-2 B11).
        destination: Output NDMI raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        nir = src.read(nir_band_index).astype("float32")
        swir1 = src.read(swir1_band_index).astype("float32")
        ndmi = (nir - swir1) / (nir + swir1 + 1e-6)
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(ndmi, 1)

    return {"status": "success", "destination": str(dst_path),
            "message": f"NDMI calculated and saved to '{dst_path}'."}


def compute_nbr(source: str, nir_band_index: int, swir2_band_index: int, destination: str) -> dict:
    """Compute NBR (Normalized Burn Ratio) and save to GeoTIFF.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B12).
        nir_band_index: Index of the near-infrared band (1-based).
        swir2_band_index: Index of the SWIR2 band (1-based, ~2.2um; Sentinel-2 B12).
        destination: Output NBR raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        nir = src.read(nir_band_index).astype("float32")
        swir2 = src.read(swir2_band_index).astype("float32")
        nbr = (nir - swir2) / (nir + swir2 + 1e-6)
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(nbr, 1)

    return {"status": "success", "destination": str(dst_path),
            "message": f"NBR calculated and saved to '{dst_path}'."}


# --- EOSS GraphRAG (ESA EO Science Strategy knowledge graph) ---

_MCP_REQUEST_TIMEOUT_SEC = 30.0
_MCP_SSE_READ_TIMEOUT_SEC = 300.0
_MCP_CONNECT_RETRIES = 3


def _create_retrying_mcp_http_client(headers=None, timeout=None, auth=None):
    """Create MCP HTTP client with retries for transient gateway/connect errors."""
    import httpx
    kwargs: dict = {
        "follow_redirects": True,
        "transport": httpx.AsyncHTTPTransport(retries=_MCP_CONNECT_RETRIES),
    }
    kwargs["timeout"] = timeout if timeout is not None else httpx.Timeout(
        _MCP_REQUEST_TIMEOUT_SEC, read=_MCP_SSE_READ_TIMEOUT_SEC,
    )
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def _build_graph_rag_toolset() -> list:
    url = os.getenv("GRAPHRAG_MCP_URL")
    auth = os.getenv("GRAPHRAG_AUTH_HEADER")
    if not url or not auth:
        raise RuntimeError("GRAPHRAG_MCP_URL and/or GRAPHRAG_AUTH_HEADER not set")
    return [McpToolset(connection_params=StreamableHTTPConnectionParams(
        url=url,
        headers={"Authorization": auth},
        timeout=_MCP_REQUEST_TIMEOUT_SEC,
        sse_read_timeout=_MCP_SSE_READ_TIMEOUT_SEC,
        httpx_client_factory=_create_retrying_mcp_http_client,
    ))]


_graph_rag_toolset = _safe_build_toolset("EOSS GraphRAG", _build_graph_rag_toolset)

# --- ArXiv (external arxiv-mcp-server package) ---
_ensure_uv_tool_installed("arxiv-mcp-server")
_ARXIV_MCP_BIN = shutil.which("arxiv-mcp-server") or str(Path.home() / ".local" / "bin" / "arxiv-mcp-server")

_arxiv_toolset = _safe_build_toolset("ArXiv", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=_ARXIV_MCP_BIN,
            args=["--storage-path", str(_ARXIV_STORAGE_DIR)],
        ),
        timeout=300,
    ))
])

# --- EO MOOCs (educational transcripts) ---
_mooc_toolset = _safe_build_toolset("MOOCs", lambda: [
    McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/esa_moocs/server.py"],
        ),
        timeout=300,
    ))
])

# --- VLM: text-only resolve_images MCP tool + self-contained describe_images below ---
_vlm_toolset = _safe_build_toolset("VLM (resolve_images)", lambda: [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/vlm_tools/server.py"],
            ),
            timeout=300,
        ),
        tool_filter=["resolve_images"],
    )
])

# Same model/endpoint as agents/subagents/vlm_init.py's `vlm` — called
# directly via litellm here since describe_images makes one self-contained
# completion, not a stateful multi-turn agent.
_VLM_MODEL = os.getenv("VLM_MODEL_NAME", "Qwen3-VL-8B-Instruct")
_VLM_ENDPOINT = os.getenv("LLAVA_ENDPOINT", "http://localhost:18001/v1")


def _encode_image_for_vlm(path) -> tuple[str, str]:
    """Return (mime_type, base64_data) for one image file, converting
    GeoTIFFs to a viewable percentile-stretched PNG first (same conversion
    load_images uses)."""
    if path.suffix.lower() in _GEOTIFF_EXTENSIONS:
        return "image/png", base64.b64encode(_geotiff_to_png_bytes(path)).decode("utf-8")
    mime = _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return mime, base64.b64encode(path.read_bytes()).decode("utf-8")


async def describe_images(path: str, question: str, limit: int = 10) -> str:
    """Ask a vision-language model a question about image(s) at a path.

    Resolves and loads the image(s) at `path` internally and sends them,
    together with `question`, to a dedicated vision-capable model —
    returning only its plain-text answer. Use this for any question about
    what an image visually shows (cloud cover, qualitative before/after
    comparison, flooding/smoke by eye, general description). Do not use it
    for quantitative raster analysis (NDVI, zonal stats, band math) — that
    stays with the GIS tools.

    Args:
        path: Absolute or relative path to a single image file (JPEG/PNG/
            GeoTIFF/...), or a directory of images. Directories are capped
            at 10 images, sorted by filename.
        question: What to ask about the image(s), e.g. "Is there visible
            cloud cover in this scene?" or "What changed between these two
            images?".
        limit: Max images to load from a directory (capped at 10).

    Returns:
        The vision model's plain-text answer, or a clear "ERROR: ..."
        string if the path resolves to no image or the model call fails.
    """
    try:
        resolved = _vlm_discover(path, limit)
    except (FileNotFoundError, ValueError) as e:
        return f"ERROR: {e}"

    content: list[dict] = [{"type": "text", "text": question}]
    for f in resolved:
        try:
            mime, b64 = _encode_image_for_vlm(f)
        except Exception as e:
            logger.warning("Failed to encode image %s for VLM: %s", f, e)
            continue
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    if len(content) == 1:
        return f"ERROR: no image at '{path}' could be loaded."

    try:
        response = await litellm.acompletion(
            model=f"openai/{_VLM_MODEL}",
            api_base=_VLM_ENDPOINT,
            api_key="EMPTY",
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        logger.warning("VLM call failed for '%s': %s", path, e)
        return f"ERROR: VLM call failed: {e}"

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Flatten every domain's toolset onto one tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = (
    _cdse_toolset
    + _climate_toolset
    + _eodashboard_toolset
    + _eumetsat_toolset
    + _gee_toolset
    + _gis_toolset
    + _graph_rag_toolset
    + _arxiv_toolset
    + _mooc_toolset
    + _vlm_toolset
    + [clip_raster_with_bbox, compute_evi, compute_ndwi, compute_ndmi, compute_nbr, describe_images]
)

logger.info("[single_agent] %d tool/toolset entries attached", len(ALL_TOOLS))

# ---------------------------------------------------------------------------
# System prompt — condensed synthesis of all 10 sub-agent instructions plus
# the orchestrator's cross-cutting routing/output/scope rules.
# ---------------------------------------------------------------------------

_SINGLE_AGENT_SYSTEM_PROMPT = ("""You are a single Earth Observation / GIS assistant with direct access to
every tool below — there is no routing layer, so you must pick the right
tool(s) yourself and chain their outputs across domains as needed.

=== CDSE (Copernicus Data Space Ecosystem) — Sentinel-1/2/3/5P, Landsat, DEM, geocoding ===
- Place name -> bbox: geocode_place (use top_result's bbox_array). Lat/lon ->
  place name: reverse_geocode_place. For macro-regions (continents, seas,
  climate zones, e.g. "the Sahel") that geocode_place can't resolve, try
  resolve_named_region next. If both fail, infer an approximate bbox
  yourself from general geographic knowledge rather than giving up.
- Pick the fetch_* tool for the requested sensor: fetch_sentinel1_grd (SAR),
  fetch_sentinel2 (optical, specify L1C/L2A), fetch_sentinel3_olci,
  fetch_sentinel3_slstr, fetch_sentinel3_syn_l2, fetch_sentinel5p_l2 (trace
  gases), fetch_landsat_ot_l1, fetch_dem (elevation).
- Sentinel-2 has no data before June 2015 — use fetch_landsat_ot_l1 instead
  for earlier dates (it's L1/TOA only; if the user needs true L2 surface
  reflectance, use GEE instead of silently substituting TOA).
- Prefer each tool's `preset` (true_color, ndvi, elevation_color, ...) over
  custom_bands or a raw evalscript when it fits.
- as_geotiff=False (JPEG quicklook, <=3 bands) only when the output is purely
  for visual inspection. Set as_geotiff=True whenever: >3 custom_bands are
  requested; the preset is a derived index/quantity (ndvi, otci, trace-gas
  value) rather than a display composite; or the result feeds anything else
  (NDVI/band math, zonal stats, reprojection, other GIS operations).
- bbox = [west, south, east, north] in WGS-84; dates = YYYY-MM-DD.

=== Climate (ERA5 reanalysis via Copernicus CDS) ===
- Discovery: list_era5_single_level_variables, list_era5_pressure_level_variables,
  list_cds_datasets. Inspect any local .nc with inspect_netcdf.
- Monthly means: fetch_era5_single_levels / fetch_era5_pressure_levels.
  Hourly: fetch_era5_hourly_single_levels / fetch_era5_hourly_pressure_levels /
  fetch_era5_land (0.1 deg, land-only — soil/runoff/high-res precip).
  Pre-aggregated daily stats: fetch_era5_daily_statistics (prefer this over
  fetching hourly + aggregating yourself when only a daily figure is needed).
  Anything else (CMIP6, seasonal forecasts, ...): fetch_cds_dataset.
- Processing: convert_hourly_to_daily (method='sum' for precipitation,
  'mean' for temperature) -> convert_daily_to_monthly -> resample_temporal;
  convert_units (K->C, m->mm, Pa->hPa, ...); clip_to_bbox.
- Always set `area` to a bbox for a regional request. Issue large requests
  as one call first; only split into smaller sequential calls if a request
  actually times out or fails.
- Do not stop once a fetch tool returns a file — check the request for every
  implied processing step (aggregation, unit conversion, clipping, index
  computation) and complete each one before answering.

=== EO Dashboard (narratives & STAC catalog) ===
- list_narratives / list_themes / search_narratives to discover narratives;
  get_narrative_content for full text; get_narrative_indicators and
  find_narratives_for_indicator to cross-link narratives <-> STAC
  collections; search_collections / get_collection_items for STAC data.
- Always report titles, themes, tags, URLs, and indicators/collections;
  state clearly (don't silently return empty) when nothing is found.

=== EUMETSAT (MTG FCI / MSG SEVIRI geostationary products only) ===
- list_eumetsat_products to look up exact product keys if unsure.
- search_eumetsat_products BEFORE download_eumetsat_product to confirm the
  match count is reasonable (download refuses >20 matches/call — narrow the
  date range/bbox/observations_per_day instead of retrying blindly).
- Use the `path` EUMETSAT actually returns for each downloaded product,
  never a path you construct yourself (extension varies by product).
- bbox = [west, south, east, north] WGS-84 (converted internally — never
  pass EUMETSAT's native order). Dates are ISO datetimes 'YYYY-MM-DDTHH:MM:SS' UTC.
- method="datastore" (fast, raw) by default; "datatailor" only when
  reprojection/clip/format conversion is explicitly needed.
- To inspect/read values from a downloaded file, call read_eumetsat_product
  (or read=true on download). Never hand a raw datastore/datatailor file to
  a GIS tool — only pass a geotiff_path from read_eumetsat_product(as_geotiff=true).
- Not for: place geocoding, Sentinel/Landsat/DEM quicklooks (use CDSE), or
  NDVI/reprojection/zonal stats (use GIS tools on its GeoTIFF output).

=== GEE (Google Earth Engine) — for collections CDSE/EUMETSAT don't cover ===
- Use only for specialized/GEE-only collections (NASA VIIRS/MODIS, JAXA
  GPM/GSMaP/ALOS, etc.) — not for imagery CDSE or EUMETSAT already serve.
- get_lat_lon to geocode a place name first if needed.
- search_gee_catalog to find the exact, current Earth Engine dataset `id`
  for anything other than these 3 safe-from-memory IDs: COPERNICUS/S2_SR_HARMONIZED
  (Sentinel-2 SR), LANDSAT/LC08/C02/T1_L2 (Landsat 8 SR), COPERNICUS/S5P/NRTI/L3_NO2.
  GEE IDs are frequently re-versioned — don't guess a memorized ID for anything else.
- Exactly one region spec per call: latitude+longitude(+buffer_m), or
  bounding_box [min_lon,min_lat,max_lon,max_lat], or region_geojson.
  Coordinates are always [longitude, latitude] order.
- Respect each dataset's temporal_extent (from the catalog search result) —
  a date range outside it silently returns zero images.
- as_geotiff=False for an 8-bit PNG quicklook (visual inspection / VLM);
  leave as_geotiff=True (default) for downstream quantitative/GIS use — GEE's
  GeoTIFF output is not viewable by the VLM tools.

=== GIS analysis (rasterio / geopandas / shapely / pyproj / pysal) ===
- Only operate on a raster/vector file already produced earlier in THIS
  conversation (by CDSE, GEE, EUMETSAT, or ERA5/climate) — never invent or
  guess a file path.
- clip_raster_with_bbox when you only have a bounding box (no shapefile);
  clip_raster_with_shapefile only if a real .shp/.geojson path already exists.
  Before clipping to an administrative boundary, confirm that boundary file
  was actually fetched/created earlier — otherwise geocode a bbox via CDSE
  and clip to that instead.
- Spectral indices (require a multi-band raster already on disk; none of
  these fetch imagery themselves; Sentinel-2 band map: B2=Blue, B3=Green,
  B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2):
    - compute_ndvi(red, nir) — general vegetation condition.
    - compute_evi(red, nir, blue, scale_factor=0.0001 for Sentinel-2 SR DN
      0-10000, else 1.0) — dense-canopy vegetation, atmosphere-corrected.
    - compute_ndwi(green, nir) — open water (McFeeters).
    - compute_ndmi(nir, swir1) — vegetation moisture / drought stress.
    - compute_nbr(nir, swir2) — burn severity; compare pre-/post-fire as dNBR.
- Also available: reprojection, raster/zonal statistics, geodesy (UTM
  zone/CRS, geodetic distance/area), map creation (create_map / create_web_map).

=== ESA EO Science Strategy knowledge graph (EOSS GraphRAG) ===
- search_eoss for natural-language questions (returns chunks + KG context).
- query_ontology for structured, read-only Cypher (MATCH ... RETURN) lookups.
- list_ontology_nodes to enumerate nodes of a label, or all node types with
  counts if no label given. Return the raw context rather than over-summarizing.

=== ArXiv (scientific literature) ===
- search_papers to discover; prefer OR'd phrasings over strict AND'd quoted
  phrases (too strict tends to zero out real results). For EO/remote-sensing/
  climate topics, useful categories are physics.ao-ph, physics.geo-ph, and
  sometimes eess.IV/eess.SP for a processing technique — omit `categories`
  entirely if unsure rather than guessing a wrong filter (e.g. never
  eess.AS or physics.data-an for EO topics, despite appearing in examples).
- Prefer get_abstract over download_paper unless full text is explicitly
  requested. Never call download_paper for more than 1 paper, or
  get_abstract for more than 5 papers, in a single task.
- If a search returns 0 results, retry ONCE with a broadened query (AND->OR,
  or drop `categories`) before reporting no papers found.

=== EO MOOCs (educational transcripts) ===
- search_moocs to find relevant course content. Return the raw retrieved
  context — do not summarize or interpret it yourself.

=== VLM image inspection (visual reasoning over files already on disk) ===
- Use describe_images(path, question) for any question about what an image
  visually shows (cloud cover, qualitative before/after comparison,
  flooding/smoke by eye, general description) — NOT for quantitative raster
  analysis (NDVI, zonal stats, band math), which stays in the GIS tools
  above. It resolves and loads the image(s) and asks a vision model
  internally, returning only its text answer — you never see raw image
  bytes yourself.
- Optionally call resolve_images on a path first if you want to confirm
  what it resolves to (e.g. how many images are in a directory) before
  calling describe_images on it.
- If comparing multiple distinct paths, call describe_images once per path
  (not all paths at once) and compare the returned descriptions yourself.

=== CROSS-CUTTING WORKFLOW RULES ===
- Multi-clause requests ("do X, then use the result to do Y"): identify every
  clause up front and do not produce a final answer until all of them are
  addressed — satisfying only the first clause is not a complete answer.
- Do not issue two calls to the same tool in the same turn with different
  parameters (e.g. a "before" and "after" image from the same sensor/tool) —
  issue them one at a time, waiting for each result, so parameters from one
  call can't bleed into the other.
- Decide up front whether imagery is for visual inspection or downstream
  quantitative/GIS use, and set as_geotiff/format accordingly on the CDSE or
  GEE fetch call — do not fetch once and try to repurpose the wrong format.
- Prefer CDSE by default for common sensor imagery (Sentinel/Landsat/DEM).
  Use EUMETSAT only for MTG FCI/MSG SEVIRI. Use GEE only for collections
  neither of those covers. Use the Climate tools only for atmospheric/
  reanalysis variables, not imagery.
- A purely narrative/text search (EO Dashboard, ArXiv, MOOCs, EOSS) needs no
  GIS or imagery tool call at all.

=== OUTPUT RULES ===
- Return tool results directly and completely, word for word — no added
  introductions, summaries, or commentary beyond what's needed to answer.
- Report local file paths as plain text, never as markdown links or URLs.
- If a tool call returns no results or fails, say so clearly instead of
  guessing, inventing data, or silently retrying more than the specific
  retry allowances noted above (e.g. arXiv's single broadened retry).

CONVERSATION: For greetings, thanks, or small talk, respond naturally and
briefly without calling any tools.

SCOPE: Only decline requests that are both (a) unrelated to Earth
Observation, satellite data, GIS, scientific literature, or ESA/space
topics, and (b) not conversational (e.g. restaurant recommendations,
financial trading, cooking, sports scores). For those, call no tools and
respond exactly: 'This question is outside the scope of the EO/GIS assistant.'"""
)


def _on_tool_error(tool, args, tool_context, error):
    """Turn a tool-execution failure into a normal tool result instead of
    letting it propagate and abort the turn (mirrors multi_agent.py)."""
    logger.warning("Tool '%s' failed: %s", getattr(tool, "name", "?"), error)
    return {"error": f"Tool '{getattr(tool, 'name', '?')}' failed: {error}"}


root = Agent(
    model=llm,
    name="single_agent",
    description=(
        "A single agent with direct access to every EO/GIS/climate/literature "
        "tool — CDSE, GEE, EUMETSAT, ERA5/Climate, GIS analysis, EO Dashboard, "
        "EOSS GraphRAG, ArXiv, MOOCs, and VLM image inspection."
    ),
    instruction=_SINGLE_AGENT_SYSTEM_PROMPT,
    tools=ALL_TOOLS,
    on_tool_error_callback=_on_tool_error,
)

APP_NAME = "EVE-AGENT-ECOSYSTEM-SINGLE"
USER_ID = "user_1"
SESSION_ID = "session_001"

session_service = InMemorySessionService()


async def init_session(app_name: str, user_id: str, session_id: str) -> InMemorySessionService:
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    logger.info("Session created: App='%s', User='%s', Session='%s'", app_name, user_id, session_id)
    return session


session = asyncio.run(init_session(APP_NAME, USER_ID, SESSION_ID))

runner = Runner(agent=root, app_name=APP_NAME, session_service=session_service)
logger.info("Runner created for agent '%s'.", runner.agent.name)


def _extract_final_text(event) -> str | None:
    if not (event.content and event.content.parts):
        return None
    for part in event.content.parts:
        text = getattr(part, "text", None)
        if text:
            return text
    return None


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    logger.info(f"\n>>> User Query: {query}")
    print(f"\n>>> User Query: {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.author:
            print(f"\n[AGENT] {event.author}")

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    print(f"  [TOOL CALL] {fc.name}({dict(fc.args)})")
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response_str = str(fr.response)
                    if len(response_str) > 300:
                        response_str = response_str[:300] + "... [truncated]"
                    print(f"  [TOOL RESULT] {fr.name} -> {response_str}")

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_response_text = text
            elif event.actions and event.actions.escalate:
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    print(f"\n<<< Agent Response: {final_response_text}")


async def get_ans_tool_path(query: str) -> str:
    """
    Runs a query through the single agent in an isolated session and returns
    a JSON string with the final answer, the leaf MCP tool calls (name +
    arguments) made to produce it, and total token usage across the run.
    Same shape as multi_agent.get_ans_tool_path, minus `agents_called` (there
    is only ever one agent here).
    """
    _orphan_guard.reset()

    session_id = f"session_{uuid.uuid4().hex}"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_answer = "Agent did not produce a final response."
    tools_called = []
    input_tokens = 0
    output_tokens = 0

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.usage_metadata:
            input_tokens += event.usage_metadata.prompt_token_count or 0
            output_tokens += event.usage_metadata.candidates_token_count or 0

        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tools_called.append(
                        {"function_name": fc.name, "arguments": dict(fc.args)}
                    )

                fr = getattr(part, "function_response", None)
                if fr is not None:
                    resp = fr.response
                    error_text = None
                    if isinstance(resp, dict):
                        if resp.get("error"):
                            error_text = str(resp["error"])
                        elif resp.get("isError"):
                            error_text = str(resp.get("content") or resp)
                    for entry in reversed(tools_called):
                        if entry["function_name"] == fr.name and "error" not in entry:
                            if error_text:
                                entry["error"] = error_text
                            break

        if event.is_final_response():
            text = _extract_final_text(event)
            if text is not None:
                final_answer = text
            elif event.actions and event.actions.escalate:
                final_answer = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )

    ended_in_error = bool(tools_called) and "error" in tools_called[-1]

    return json.dumps({
        "final_answer": final_answer,
        "ended_in_error": ended_in_error,
        "tools_called": tools_called,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }, default=str)


async def run_conversation():
    await call_agent_async(
        "Search arXiv for papers on sea-level rise monitoring using satellite altimetry published since 2024 and return up to 15 results.",
        runner, USER_ID, SESSION_ID,
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
