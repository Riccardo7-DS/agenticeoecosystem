"""
MCP server exposing:
  1. Each Sentinel Hub collection as a separate tool (CDSE Process API).
  2. Geocoding tools (place name <-> bbox / lat-lon) via OpenStreetMap Nominatim.

Run with:
    python server.py

Or via stdio transport (for use in MCP clients / Claude Desktop):
    {
        "command": "python",
        "args": ["/path/to/server.py", "--transport", "stdio"],
        "env": {
            "CDSE_CLIENT_ID": "...",
            "CDSE_CLIENT_SECRET": "...",
            "CDSE_OUTPUT_DIR": "/path/to/save/images"
        }
    }

Image-return behaviour
-----------------------
By default, fetch_* tools DO NOT return the image bytes inline. Instead the
image is written to disk (CDSE_OUTPUT_DIR, or an explicit save_path=...) and
the tool returns only a TextContent block with JSON metadata, including
`saved_path`. This keeps responses small and avoids pushing large base64
blobs through the MCP transport / model context.

Pass return_image=True on any fetch_* tool to additionally get an
ImageContent block with the base64 image inline (old behaviour), on top of
the file still being saved to disk.

GeoTIFF support
-----------------
Every fetch_* tool now accepts as_geotiff=True, which:
  - saves a georeferenced GeoTIFF (CRS/transform embedded from bbox) with
    raw, analysis-ready FLOAT32 values instead of an 8-bit display JPEG.
  - automatically picks a ".tif" default filename extension instead of
    ".jpg" when save_path isn't explicitly given.
Use as_geotiff=True whenever the output will feed a raster-analysis tool
(metadata_raster, raster_band_statistics, reproject_raster, ...) rather
than being viewed directly.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
import uuid
from typing import Annotated, Any, Optional

import httpx
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from agenticeoecosystem.servers.cdse_geocode.helper import (
    fetch_sentinel1_grd,
    fectch_sentinel2,
    fetch_sentinel3_olci,
    fetch_sentinel3_slstr,
    fetch_sentinel3_syn_l2,
    fetch_sentinel5p_l2,
    fetch_landsat_ot_l1,
    fetch_dem,
)

from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sentinel-hub-mcp")

mcp = FastMCP(
    "sentinel-hub",
    instructions=(
        "Sentinel Hub satellite imagery tools, plus geocoding helpers. "
        "Each fetch_* tool retrieves a clipped AOI image for a specific satellite "
        "collection and SAVES IT TO DISK by default (see `saved_path` in the "
        "returned metadata) instead of returning the image inline — pass "
        "return_image=true to also get the base64 image in the response. "
        "Pass as_geotiff=true to save a georeferenced GeoTIFF with raw "
        "FLOAT32 analysis-ready values instead of an 8-bit display JPEG. "
        "Always pass bbox as [west, south, east, north] in WGS-84 decimal degrees. "
        "Dates are ISO-8601 strings: YYYY-MM-DD. "
        "Use geocode_place to turn a place name into a bbox before calling a "
        "fetch_* tool, or reverse_geocode_place to turn lat/lon into a place name."
    ),
    host="0.0.0.0", port=8000, stateless_http=True
)

# ---------------------------------------------------------------------------
# Output directory / default save-path handling
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path(
    os.environ.get("CDSE_OUTPUT_DIR", str(ROOT_DIR / "tool_output" / "cdse_geocode"))
)


def _default_save_path(collection: str, ext: str = "jpg") -> str:
    """Build a unique default file path for a given collection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    fname = f"{collection}_{ts}_{uid}.{ext}"
    return str(OUTPUT_DIR / fname)


def _to_content(result: dict, return_image: bool = False) -> list[ImageContent | TextContent]:
    """
    Convert a sentinel_hub fetch result dict into MCP content blocks.

    Always returns:
      - TextContent with JSON metadata (collection, dims, format, saved_path,
        evalscript)

    Only returns (in addition), when return_image=True:
      - ImageContent with the image bytes as base64
    """
    meta = {
        "collection":  result["collection"],
        "width_px":    result["width"],
        "height_px":   result["height"],
        "media_type":  result["media_type"],
        "saved_path":  result["saved_path"],
        "evalscript":  result["evalscript"],
    }
    text_block = TextContent(
        type="text",
        text=json.dumps(meta, indent=2),
    )

    if not return_image:
        return [text_block]

    image_block = ImageContent(
        type="image",
        data=result["image_b64"],
        mimeType=result["media_type"],
    )
    return [image_block, text_block]


# ---------------------------------------------------------------------------
# Tool: Sentinel-1 GRD
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel1_grd",
    description=(
        "Fetch a Sentinel-1 GRD SAR image. Saved to disk by default (see save_path/"
        "return_image). "
        "Presets: 'vv_vh_rgb' (pseudo-RGB from VV+VH, dual-pol IW mode), "
        "'hh_hv_rgb' (pseudo-RGB from HH+HV, dual-pol EW mode), "
        "Use custom_bands to select any polarisation(s) manually. "
        "back_coeff options: GAMMA0_ELLIPSOID (default), GAMMA0_TERRAIN (RTC), SIGMA0_ELLIPSOID. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "linear backscatter values per polarisation (instead of an 8-bit "
        "display JPEG with dB scaling/RGB compositing) for feeding into "
        "raster analysis tools (metadata_raster, raster_band_statistics, ...). "
        "Hard output limits: width/height <= 2500 px each, and ground sample "
        "distance (bbox size / width or height) <= ~1500 m/pixel — for a large "
        "bbox, raise width/height accordingly or shrink the bbox."
    ),
)
def tool_sentinel1_grd(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "One of: vv_vh_rgb | hh_hv_rgb "
        "Omit if supplying custom_bands or evalscript. Defaults to vv_vh_rgb.",
    ] = "vv_vh_rgb",
    custom_bands: Annotated[
        Optional[list[str]],
        "Explicit polarisation list, e.g. ['VV'] or ['VV','VH']. "
        "Overrides preset.",
    ] = None,
    evalscript: Annotated[
        Optional[str],
        "Raw Sentinel Hub evalscript (VERSION=3 JS). Overrides preset and custom_bands.",
    ] = None,
    orthorectify: Annotated[bool, "Apply geometric terrain correction (default True)"] = True,
    back_coeff: Annotated[
        str,
        "Backscatter coefficient: GAMMA0_ELLIPSOID | GAMMA0_TERRAIN | SIGMA0_ELLIPSOID",
    ] = "GAMMA0_ELLIPSOID",
    dem_instance: Annotated[
        Optional[str],
        "DEM for orthorectification, e.g. 'COPERNICUS_30'",
    ] = None,
    acquisition_mode: Annotated[
        Optional[str],
        "Filter by mode: IW | EW | SM | WV",
    ] = None,
    polarization: Annotated[
        Optional[str],
        "Filter by polarization scheme: DV | DH | SV | SH",
    ] = None,
    orbit_direction: Annotated[
        Optional[str],
        "Filter by orbit: ASCENDING | DESCENDING",
    ] = None,
    resolution: Annotated[Optional[str], "HIGH | MEDIUM"] = None,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 linear "
        "backscatter values instead of an 8-bit display JPEG. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel1_grd", ext="tif" if as_geotiff else "jpg")
    result = fetch_sentinel1_grd(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        orthorectify=orthorectify,
        back_coeff=back_coeff,
        dem_instance=dem_instance,
        acquisition_mode=acquisition_mode,
        polarization=polarization,
        orbit_direction=orbit_direction,
        resolution=resolution,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Sentinel-2
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel2",
    description=(
        "Fetch a Sentinel-2 image at ~10–60 m resolution. Saved to disk by default "
        "(see save_path/return_image). "
        "Collection : choose form L2A or L1c, defaults to L2A"
        "Preset: 'true_color' (R=B04, G=B03, B=B02). 'ndvi' (NIR=B08, R=B04) "
        "Use custom_bands for any combination of B01–B12, B8A. "
        "Note: no atmospheric correction — use L2A for surface reflectance. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "values (instead of an 8-bit display JPEG) for feeding into raster "
        "analysis tools (metadata_raster, raster_band_statistics, reproject_raster, ...). "
        "Hard output limits: width/height <= 2500 px each, and ground sample "
        "distance (bbox size / width or height) <= ~1500 m/pixel — for a large "
        "bbox, raise width/height accordingly or shrink the bbox."
    ),
)
def tool_sentinel2(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    collection: Annotated[str, "The collection of Sentinel-2 L2A or L1C"],
    preset: Annotated[
        Optional[str],
        "true_color. Omit if supplying custom_bands or evalscript. Defaults to true_color.",
    ] = "true_color",
    custom_bands: Annotated[
        Optional[list[str]],
        "Band list, e.g. ['B08','B04','B03'] for NIR false colour. "
        "Available: B01–B12, B8A.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    max_cloud_cover: Annotated[
        Optional[float], "Maximum cloud coverage 0–100 (%)"
    ] = None,
    mosaicking_order: Annotated[
        str, "leastCC | mostRecent | leastRecent"
    ] = "leastCC",
    harmonize_values: Annotated[
        bool, "Apply SAFE/BASELINE processing baseline harmonisation"
    ] = False,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 values instead "
        "of an 8-bit display JPEG — use when the output feeds a raster analysis "
        "tool rather than being viewed directly. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel2", ext="tif" if as_geotiff else "jpg")
    result = fectch_sentinel2(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collection=collection,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        max_cloud_cover=max_cloud_cover,
        mosaicking_order=mosaicking_order,
        harmonize_values=harmonize_values,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Sentinel-3 OLCI
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel3_olci",
    description=(
        "Fetch a Sentinel-3 OLCI ocean/land colour image at ~300 m resolution. "
        "Saved to disk by default (see save_path/return_image). "
        "21 spectral bands from 400 nm (B01) to 1020 nm (B21). "
        "Preset: 'true_color' (B08 red / B06 green / B04 blue), 'otci' (Terrestrial "
        "Chlorophyll Index colour ramp). "
        "Good for large-area overviews, ocean colour, vegetation at continental scale. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "reflectance (or raw OTCI index) values instead of an 8-bit display JPEG "
        "for feeding into raster analysis tools."
    ),
)
def tool_sentinel3_olci(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "true_color or otci. Omit if supplying custom_bands or evalscript. "
        "Defaults to true_color.",
    ] = "true_color",
    custom_bands: Annotated[
        Optional[list[str]],
        "Band list from B01–B21, e.g. ['B17','B08','B04'] for NIR false colour.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 values instead "
        "of an 8-bit display JPEG. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel3_olci", ext="tif" if as_geotiff else "jpg")
    result = fetch_sentinel3_olci(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Sentinel-3 SLSTR
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel3_slstr",
    description=(
        "Fetch a Sentinel-3 SLSTR image. Saved to disk by default (see save_path/"
        "return_image). "
        "Shortwave bands S1–S6 (0.55–2.25 µm) at 500 m–1 km; "
        "thermal bands S7–S9, F1–F2 (3.7–12 µm). "
        "Preset: 'false_color' (S3n/S2n/S1n approximate visible composite) or ndvi. "
        "Append the view suffix to custom band names (e.g. S2n, S2o), "
        "or omit suffix and set view= to apply it automatically. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "values instead of an 8-bit display JPEG."
    ),
)
def tool_sentinel3_slstr(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "false_color or ndvi. Omit if supplying custom_bands or evalscript. "
        "Defaults to false_color.",
    ] = "false_color",
    custom_bands: Annotated[
        Optional[list[str]],
        "Bands without view suffix, e.g. ['S5','S3','S1']. "
        "The view= parameter appends 'n' or 'o' automatically.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 values instead "
        "of an 8-bit display JPEG. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel3_slstr", ext="tif" if as_geotiff else "jpg")
    result = fetch_sentinel3_slstr(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Sentinel-3 SYN L2
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel3_syn_l2",
    description=(
        "Fetch a Sentinel-3 SYN L2 (Synergy OLCI+SLSTR combined) image at ~300 m. "
        "Saved to disk by default (see save_path/return_image). "
        "Surface reflectance product. "
        "OLCI channels: B01–B12, B16-B18, B21. SLSTR channels: S1–S6. "
        "Preset: 'true_color' (B08 / B06 / B04), 'ndvi'. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "values instead of an 8-bit display JPEG."
    ),
)
def tool_sentinel3_syn_l2(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "true_color or ndvi. Omit if supplying custom_bands or evalscript. "
        "Defaults to true_color.",
    ] = "true_color",
    custom_bands: Annotated[
        Optional[list[str]],
        "Band list, e.g. ['B17','B08','B04'] for NIR false colour.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 values instead "
        "of an 8-bit display JPEG. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel3_syn_l2", ext="tif" if as_geotiff else "jpg")
    result = fetch_sentinel3_syn_l2(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Sentinel-5P L2
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_sentinel5p_l2",
    description=(
        "Fetch a Sentinel-5P TROPOMI L2 trace-gas product at ~3.5 × 5.5 km resolution. "
        "Saved to disk by default (see save_path/return_image). "
        "Each band is a geophysical quantity (mol/m², DU, etc.), visualised as greyscale "
        "by default. "
        "Presets: no2 | co | o3 | so2 | ch4 | hcho | aer_ai_340_380 | cloud_fraction. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with the raw, unscaled "
        "physical value as FLOAT32 (instead of the 0-1 clamped/scaled display "
        "greyscale) — required for any quantitative use of the values."
    ),
)
def tool_sentinel5p_l2(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "no2 | co | o3 | so2 | ch4 | hcho | aer_ai_340_380 | cloud_fraction. "
        "Omit if supplying custom_bands or evalscript. Defaults to no2.",
    ] = "no2",
    custom_bands: Annotated[
        Optional[list[str]],
        "Single S5P band name list, e.g. ['NO2'].",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with the raw unscaled FLOAT32 "
        "physical value instead of the 0-1 clamped display greyscale JPEG. "
        "Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("sentinel5p_l2", ext="tif" if as_geotiff else "jpg")
    result = fetch_sentinel5p_l2(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: Landsat OT L1
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_landsat_ot_l1",
    description=(
        "Fetch a Landsat 8/9 OLI-TIRS L1 image at ~30 m (15 m panchromatic). "
        "Saved to disk by default (see save_path/return_image). "
        "Preset: 'true_color' (R=B04, G=B03, B=B02). "
        "custom_bands: B01 (coastal), B02 (blue), B03 (green), B04 (red), "
        "B05 (NIR), B06 (SWIR-1), B07 (SWIR-2), B08 (pan), "
        "B09 (cirrus), B10/B11 (thermal). "
        "e.g. ['B07','B05','B04'] for SWIR false colour. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "values (instead of an 8-bit display JPEG) for feeding into raster "
        "analysis tools (metadata_raster, raster_band_statistics, ...)."
    ),
)
def tool_landsat_ot_l1(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    start_date: Annotated[str, "Start of date range, YYYY-MM-DD"],
    end_date: Annotated[str, "End of date range, YYYY-MM-DD"],
    preset: Annotated[
        Optional[str],
        "true_color. Omit if supplying custom_bands or evalscript. Defaults to true_color.",
    ] = "true_color",
    custom_bands: Annotated[
        Optional[list[str]],
        "Band list, e.g. ['B05','B04','B03'] for NIR false colour.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    max_cloud_cover: Annotated[
        Optional[float], "Maximum cloud coverage 0–100 (%)"
    ] = None,
    mosaicking_order: Annotated[
        str, "leastCC | mostRecent | leastRecent"
    ] = "leastCC",
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 values instead "
        "of an 8-bit display JPEG — use when the output feeds a raster analysis "
        "tool rather than being viewed directly. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("landsat_ot_l1", ext="tif" if as_geotiff else "jpg")
    result = fetch_landsat_ot_l1(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        max_cloud_cover=max_cloud_cover,
        mosaicking_order=mosaicking_order,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Tool: DEM
# ---------------------------------------------------------------------------

@mcp.tool(
    name="fetch_dem",
    description=(
        "Fetch a Digital Elevation Model image. Saved to disk by default (see "
        "save_path/return_image). Static product — date range is required "
        "by the API but does not filter scenes. "
        "Presets: 'elevation_grey' (0–8000 m greyscale), "
        "'elevation_color' (blue→green→yellow→red ramp, 0–5000 m). "
        "dem_instance: COPERNICUS_30 (30 m, default) | COPERNICUS_90 (90 m) | MAPZEN. "
        "Set as_geotiff=True to get a georeferenced GeoTIFF with raw FLOAT32 "
        "elevation values in metres (ignores the display presets) for feeding "
        "into raster analysis tools (metadata_raster, raster_band_statistics, ...)."
    ),
)
def tool_dem(
    bbox: Annotated[list[float], "Bounding box [west, south, east, north] in WGS-84"],
    preset: Annotated[
        Optional[str],
        "elevation_grey | elevation_color. Omit if supplying custom_bands or evalscript. "
        "Defaults to elevation_grey.",
    ] = "elevation_grey",
    custom_bands: Annotated[
        Optional[list[str]],
        "Typically ['DEM'] for raw elevation values.",
    ] = None,
    evalscript: Annotated[Optional[str], "Raw evalscript. Overrides everything."] = None,
    dem_instance: Annotated[
        str, "COPERNICUS_30 | COPERNICUS_90 | MAPZEN"
    ] = "COPERNICUS_30",
    start_date: Annotated[str, "Ignored by DEM but required by API (YYYY-MM-DD)"] = "2020-01-01",
    end_date: Annotated[str, "Ignored by DEM but required by API (YYYY-MM-DD)"] = "2020-12-31",
    width: Annotated[int, "Output image width in pixels"] = 512,
    height: Annotated[int, "Output image height in pixels"] = 512,
    save_path: Annotated[
        Optional[str],
        "Local file path to save the image. Defaults to an auto-generated path "
        "under CDSE_OUTPUT_DIR if omitted.",
    ] = None,
    return_image: Annotated[
        bool,
        "If True, also include the raw base64 image inline in the response "
        "(in addition to saving it to disk). Default False.",
    ] = False,
    as_geotiff: Annotated[
        bool,
        "If True, save a georeferenced GeoTIFF with raw FLOAT32 elevation values "
        "(metres) instead of an 8-bit display JPEG, ignoring preset — use when "
        "the output feeds a raster analysis tool. Default False.",
    ] = False,
) -> list[ImageContent | TextContent]:
    save_path = save_path or _default_save_path("dem", ext="tif" if as_geotiff else "jpg")
    result = fetch_dem(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        preset=preset,
        custom_bands=custom_bands,
        evalscript=evalscript,
        dem_instance=dem_instance,
        width=width,
        height=height,
        save_path=save_path,
        as_geotiff=as_geotiff,
    )
    return _to_content(result, return_image=return_image)


# ---------------------------------------------------------------------------
# Geocoding tools (OpenStreetMap Nominatim)
# ---------------------------------------------------------------------------

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GeocodeMCPServer"
GEOCODE_TIMEOUT = 60

geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=GEOCODE_TIMEOUT)


async def _http_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Perform an async HTTP GET with sensible defaults."""
    async with httpx.AsyncClient(timeout=GEOCODE_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp


@mcp.tool()
async def geocode_place(
    place_name: str,
    buffer_km: float = 0.0,
    limit: int = 5,
) -> str:
    """
    Convert a place name to a bounding box.

    Geocodes a place name (city, region, country, landmark, etc.) into a
    bounding box string in "west,south,east,north" format.

    Uses the OpenStreetMap Nominatim API (free, no API key required).

    Args:
        place_name: Name of the place to geocode (e.g. "Greece",
                    "Athens", "Evia island", "Peloponnese").
        buffer_km:  Optional buffer in km to expand the bbox (default 0).
        limit:      Maximum number of candidate results to return (default 5).

    Returns:
        JSON with the top result's bbox string (ready to pass to other tools)
        and all candidate matches. The `bbox_array` field on each result can
        be passed straight into any fetch_* tool's `bbox` argument.
    """
    logger.info("Geocoding place: %s", place_name)

    try:
        resp = await _http_get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": place_name,
                "format": "json",
                "limit": str(limit),
                "addressdetails": "1",
            },
            headers={"User-Agent": NOMINATIM_USER_AGENT},
        )
        results = resp.json()
    except Exception as exc:
        return json.dumps({"error": f"Geocoding failed: {exc}"})

    if not results:
        return json.dumps(
            {
                "error": f"No results found for '{place_name}'.",
                "place_name": place_name,
            }
        )

    buf = buffer_km / 111.0

    def _parse_result(r: dict) -> dict:
        # Nominatim boundingbox is [south, north, west, east]
        bb = r.get("boundingbox", [])
        south, north, west, east = (
            float(bb[0]) - buf,
            float(bb[1]) + buf,
            float(bb[2]) - buf,
            float(bb[3]) + buf,
        )
        return {
            "display_name": r.get("display_name", ""),
            "bbox": f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}",
            "bbox_array": [
                round(west, 4),
                round(south, 4),
                round(east, 4),
                round(north, 4),
            ],
            "lat": float(r.get("lat", 0)),
            "lon": float(r.get("lon", 0)),
            "osm_type": r.get("osm_type", ""),
            "class": r.get("class", ""),
            "type": r.get("type", ""),
        }

    parsed = [_parse_result(r) for r in results]

    return json.dumps(
        {
            "place_name": place_name,
            "top_result": parsed[0],
            "all_results": parsed,
            "total_results": len(parsed),
        }
    )


@mcp.tool()
async def reverse_geocode_place(
    latitude: float,
    longitude: float,
    zoom: int = 18,
) -> str:
    """
    Convert latitude and longitude to a place name (reverse geocoding).

    Uses the OpenStreetMap Nominatim API via geopy.

    Args:
        latitude:  Latitude coordinate.
        longitude: Longitude coordinate.
        zoom:      Detail level from 0 (country) to 18 (building).

    Returns:
        JSON with the place name and address details.
    """
    logger.info("Reverse geocoding: %s, %s", latitude, longitude)

    if not 0 <= zoom <= 18:
        return json.dumps(
            {
                "error": "Zoom must be between 0 and 18.",
                "zoom": zoom,
            }
        )

    try:
        location = geolocator.reverse(f"{latitude}, {longitude}", zoom=zoom)
    except Exception as exc:
        return json.dumps({"error": f"Reverse geocoding failed: {exc}"})

    return json.dumps(
        {
            "latitude": latitude,
            "longitude": longitude,
            "address": location.address,
        }
    )


# ---------------------------------------------------------------------------
# Named-region gazetteer (fixed lookup, not live geocoding)
#
# Covers broad macro-regions (continents, seas, multi-country areas) that
# geocode_place (a place/administrative-boundary geocoder) doesn't resolve
# well — there's no single OSM boundary for "Europe" or "the Sahel". These
# are commonly-used EO/climate-community region definitions, not official
# standardized boundaries.
# ---------------------------------------------------------------------------

_NAMED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name -> (west, south, east, north)
    "europe": (-25.0, 34.0, 45.0, 72.0),
    "north atlantic": (-80.0, 0.0, 20.0, 70.0),
    "middle east": (25.0, 12.0, 63.0, 42.0),
    "mediterranean": (-6.0, 30.0, 37.0, 46.0),
    "iberian peninsula": (-9.5, 36.0, 4.5, 44.0),
    "italian peninsula": (6.6, 35.5, 18.6, 47.1),
    "central europe": (5.0, 45.0, 24.0, 55.0),
    "southern europe": (-10.0, 34.0, 30.0, 46.0),
    "sahel": (-18.0, 10.0, 40.0, 20.0),
    "alps": (5.5, 43.5, 16.5, 48.0),
    "swiss alps": (5.9, 45.8, 10.5, 47.8),
    "north sea": (-4.0, 51.0, 9.0, 61.0),
    "sub-saharan africa": (-18.0, -35.0, 52.0, 20.0),
    "uk and ireland": (-11.0, 49.0, 2.0, 61.0),
    "spain and portugal": (-9.5, 36.0, 4.5, 44.0),
    "congo basin": (12.0, -6.0, 31.0, 8.0),
    "po valley": (7.5, 44.5, 12.5, 45.7),
    "northern hemisphere": (-180.0, 0.0, 180.0, 90.0),
}


@mcp.tool()
def resolve_named_region(region_name: str) -> str:
    """
    Look up a bounding box for a well-known named macro-region (continent,
    sea, or multi-country area) that isn't a single administrative unit a
    place geocoder can resolve — e.g. "Europe", "the Sahel", "Middle East",
    "Iberian Peninsula". For a specific city, landmark, or single country,
    use geocode_place instead.

    This is a small fixed lookup table (not a live geocoding call), covering
    commonly-referenced EO/climate-community region definitions rather than
    official standardized boundaries.

    Bbox format matches geocode_place: "west,south,east,north". Note the
    climate server's `area` parameter uses a DIFFERENT order,
    [north, west, south, east] — reorder before passing into climate tools.

    Args:
        region_name: Name of the region, e.g. "Europe", "North Atlantic",
                     "Middle East", "the Sahel", "Iberian Peninsula".

    Returns:
        JSON with the matched region name and its bbox, or an error listing
        available region names if no match was found.
    """
    query = region_name.strip().lower()
    if query.startswith("the "):
        query = query[4:]

    match_key = None
    if query in _NAMED_REGIONS:
        match_key = query
    else:
        for key in _NAMED_REGIONS:
            if key in query or query in key:
                match_key = key
                break

    if match_key is None:
        return json.dumps(
            {
                "error": f"No known region matching '{region_name}'.",
                "available_regions": sorted(_NAMED_REGIONS.keys()),
            }
        )

    west, south, east, north = _NAMED_REGIONS[match_key]
    return json.dumps(
        {
            "region_name": match_key,
            "bbox": f"{west},{south},{east},{north}",
            "bbox_array": [west, south, east, north],
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDSE + Geocode MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")