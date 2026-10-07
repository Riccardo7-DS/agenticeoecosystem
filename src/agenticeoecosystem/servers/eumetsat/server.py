"""
MCP server exposing EUMETSAT geostationary product search & download
(MTG FCI, MSG SEVIRI) via the `eumetsearch` package.

Scope is intentionally narrow: this server owns *search and download of raw /
tailored EUMETSAT Data Store products* only. It deliberately does not
duplicate capabilities that already live elsewhere in this ecosystem:
  - NDVI, reprojection, clipping, zonal stats  -> gis-mcp
  - Sentinel/Landsat/DEM quicklook imagery      -> cdse_geocode
  - Place name <-> bbox geocoding               -> gee-mcp / cdse_geocode

Bounding boxes on this server's tools use [west, south, east, north] in
WGS-84 decimal degrees, matching the convention used by cdse_geocode and
gis-mcp. This is converted internally to eumdac's native [north, south,
west, east] order.

Run with:
    python server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from agenticeoecosystem.utils.output_paths import server_output_dir
from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - EUMETSAT-MCP - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from eumetsearch import EUMDownloader, products_list  # noqa: E402

# eumetsearch registers a global dask ProgressBar on import (writes raw
# "[####] | N% Completed" text to stdout on every dask computation, e.g. when
# satpy loads a band). Harmless for eumetsearch's own CLI use, but this
# server's stdout is the MCP JSON-RPC stream to its client — every line must
# be exactly one JSON-RPC message, so that progress text corrupts the
# protocol. Drop it here rather than in eumetsearch itself, since the
# progress bar is still wanted there.
from dask.callbacks import Callback  # noqa: E402

Callback.active.clear()

app = FastMCP("eumetsat-mcp")

OUTPUT_DIR = server_output_dir("eumetsat", "EUMETSAT_OUTPUT_DIR")

# Keep single tool calls fast and bounded — large batch jobs belong in
# eumetsearch's own CLI pipeline (main.py / pipeline_data.py), not here.
MAX_PRODUCTS_PER_CALL = 20


def _wsen_to_nswe(bbox: Optional[list[float]]) -> Optional[list[float]]:
    """Convert [west, south, east, north] (this server's public convention)
    to eumdac's native [north, south, west, east]."""
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must be exactly [west, south, east, north].")
    west, south, east, north = bbox
    return [north, south, west, east]


def _new_downloader(product: str, format: str = "netcdf4") -> EUMDownloader:
    if product not in products_list:
        raise ValueError(
            f"Unknown product '{product}'. Call list_eumetsat_products for valid keys."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    product_id = products_list[product]["product_id"]
    return EUMDownloader(
        product_id=product_id,
        output_dir=str(OUTPUT_DIR),
        format=format,
    )


# Sub-satellite longitude for each geostationary product, used as the
# fallback center-of-view when a request has no bbox (full disk). Prime
# services sit at 0°E; IODC and Rapid Scan are offset.
_PRODUCT_SUBSATELLITE_LON = {
    "MTG-FCI-L1C-FDHSI": 0.0,
    "MTG-FCI-L1C-HRFI": 0.0,
    "MTG-FCI-L2-CLM": 0.0,
    "MTG-FCI-L2-AFM": 0.0,
    "MTG-FCI-L2-OCA": 0.0,
    "MTG-FCI-L2-GII": 0.0,
    "MTG-FCI-L2-AMV": 0.0,
    "MSG-SEVIRI-L1-HRY": 0.0,
    "MSG-SEVIRI-L1-IODC": 41.5,
    "MSG-SEVIRI-RSS": 3.5,
}


def _solar_noon_utc_hour(product: str, bbox: Optional[list[float]]) -> int:
    """Approximate the UTC hour of local solar noon for a request.

    Geostationary products (MTG FCI, MSG SEVIRI) repeat every 5-15 minutes,
    so a multi-day download with no hour specified would otherwise pull in
    every slot of every day. Solar noon is when the sun is highest over the
    target location, which maximises reflectance/illumination for VIS/NIR
    channels and is the single most useful slot to default to. Longitude is
    taken from the bbox center when given, otherwise from the satellite's
    sub-satellite point (its natural center of view for a full-disk request).
    Uses the simple lon/15 approximation (ignores the equation of time and
    longitude, which shift solar noon by at most ~15-20 minutes).
    """
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be exactly [west, south, east, north].")
        west, _south, east, _north = bbox
        lon = (west + east) / 2.0
    else:
        lon = _PRODUCT_SUBSATELLITE_LON.get(product, 0.0)
    return round(12 - lon / 15.0) % 24


# eumetsearch's products_list tags SEVIRI products with the HRIT segment
# reader, but Data Store downloads of those products are native-format
# (.nat) archives, which that reader can't parse — fall back to the reader
# that actually matches what's on disk.
_READER_FALLBACKS = {
    "seviri_l1b_hrit": "seviri_l1b_native",
}

# Product archives can extract non-imagery sidecar files alongside the real
# data — JPEG quicklooks, XML manifests/metadata, and (for chunked products
# like MTG FCI) a CHK-TRAIL scan-summary chunk as opposed to the CHK-BODY
# image chunks. satpy doesn't know how to open these, and handing them to it
# anyway doesn't just fail cleanly — it has been observed to segfault the
# reader rather than raise a catchable Python exception. Filter to files
# satpy's readers are actually meant to consume before ever reaching satpy.
_READABLE_EXTENSIONS = {".nc", ".nat"}


def _is_readable_data_file(path: Path) -> bool:
    if path.suffix.lower() not in _READABLE_EXTENSIONS:
        return False
    if "CHK-TRAIL" in path.name:
        return False
    return True


def _downloaded_product_paths(identifier: Optional[str], files: list[str], product_count: int) -> list[str]:
    """Attribute chunked files only when one product makes ownership unambiguous."""
    if product_count == 1:
        return list(files)
    return [path for path in files if identifier and identifier in Path(path).name]


def _extract_leftover_zips(directory: Path) -> None:
    """Extract and remove any .zip files sitting in `directory`.

    'datastore' downloads are already unzipped by EUMDownloader, but this is
    a defensive pass so read_eumetsat_product also works after a
    'datatailor' download or a partially-processed directory.
    """
    for zip_path in directory.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(directory)
            zip_path.unlink()
        except zipfile.BadZipFile:
            logger.warning(f"Skipping corrupt zip file {zip_path}")


@app.tool()
def list_eumetsat_products() -> str:
    """List every EUMETSAT product this server can search/download.

    Covers MTG FCI (Level 1C FDHSI/HRFI, Level 2 cloud mask/fire/OCA) and
    MSG SEVIRI (Prime, IODC, Rapid Scan). Returns a JSON dict keyed by product
    name — pass that key as `product` to search_eumetsat_products /
    download_eumetsat_product — with the underlying EUMETSAT product ID, the
    satpy reader used to open downloaded files, and the available channel
    names.
    """
    return json.dumps(
        {
            key: {
                "product_id": meta["product_id"],
                "satpy_reader": meta.get("satpy_reader"),
                "bands": meta.get("bands"),
            }
            for key, meta in products_list.items()
        },
        indent=2,
    )


@app.tool()
def search_eumetsat_products(
    product: str,
    start_date: str,
    end_date: str,
    bbox: Optional[list[float]] = None,
) -> str:
    """Search for EUMETSAT products in a time window without downloading anything.

    Use this to check how many products a request would match before calling
    download_eumetsat_product, which refuses ranges above the per-call limit.
    Geostationary products (MTG FCI, MSG SEVIRI) repeat every 5-15 minutes, so
    a wide date range can match far more products than are returned here — the
    listing itself is capped at MAX_PRODUCTS_PER_CALL entries; check "count"
    (always accurate) and narrow start_date/end_date or add a bbox if it's
    larger than the number of entries actually returned.

    Args:
        product: product key from list_eumetsat_products, e.g. "MTG-FCI-L1C-FDHSI".
        start_date, end_date: ISO datetime strings 'YYYY-MM-DDTHH:MM:SS' (UTC).
        bbox: optional [west, south, east, north] bounding box in decimal
              degrees. Omit for the full disk.

    Returns:
        JSON with the true "count", a "products" list of up to
        MAX_PRODUCTS_PER_CALL matching {"identifier", "date"} entries, and
        "truncated" (true if "count" exceeds how many "products" are listed).
    """
    downloader = _new_downloader(product)
    result = downloader.search(
        start_time=start_date, end_time=end_date, bounding_box=_wsen_to_nswe(bbox)
    )
    products = result.get("products", [])
    truncated = len(products) > MAX_PRODUCTS_PER_CALL
    result["products"] = products[:MAX_PRODUCTS_PER_CALL]
    result["truncated"] = truncated
    if truncated:
        result["note"] = (
            f"Only the first {MAX_PRODUCTS_PER_CALL} of {result.get('count', len(products))} "
            "matching products are listed. Narrow start_date/end_date or add a bbox "
            "to see the rest."
        )
    return json.dumps(result, indent=2)


class DownloadParams(BaseModel):
    product: str = Field(
        ..., description="Product key from list_eumetsat_products, e.g. 'MTG-FCI-L1C-FDHSI'."
    )
    start_date: str = Field(..., description="Start datetime 'YYYY-MM-DDTHH:MM:SS' (UTC).")
    end_date: str = Field(..., description="End datetime 'YYYY-MM-DDTHH:MM:SS' (UTC).")
    bbox: Optional[list[float]] = Field(
        None,
        description="[west, south, east, north] bounding box in decimal degrees. Omit for full disk.",
    )
    format: str = Field(
        "netcdf4",
        description=(
            "'netcdf4' or 'geotiff'. Only takes effect with method='datatailor' — "
            "'datastore' downloads always return the archived format."
        ),
    )
    method: str = Field(
        "datastore",
        description=(
            "'datastore' = direct raw file download (fast, no server-side processing). "
            "'datatailor' = EUMETSAT-side reprojection to geographic lat/lon, clipped to "
            "bbox, converted to `format` (slower — polls until the customisation finishes). "
            "Prefer 'datatailor' when you need the data clipped/reprojected instead of the "
            "full native geostationary disk."
        ),
    )
    observations_per_day: int = Field(
        1, description="Number of observation slots to fetch per day in the range."
    )
    start_hour: Optional[int] = Field(
        None,
        description=(
            "UTC hour of the first observation each day. Omit to auto-select the UTC "
            "hour of local solar noon for the request's location (bbox center, or the "
            "satellite's sub-satellite point for full-disk requests) — this maximises "
            "reflectance/illumination for VIS/NIR channels and avoids pulling every "
            "5-15 minute repeat-cycle slot in the day range. The response's `warning` "
            "field reports the hour that was chosen."
        ),
    )
    interval_minutes: int = Field(10, description="Length of each observation slot in minutes.")
    read: bool = Field(
        False,
        description=(
            "If true, read the downloaded files with satpy immediately after download "
            "and include per-band summary stats in the response, equivalent to calling "
            "read_eumetsat_product afterwards."
        ),
    )


@app.tool(
    description=(
        "Download EUMETSAT MTG FCI / MSG SEVIRI product files for a time window. "
        "Files are saved to disk under EUMETSAT_OUTPUT_DIR and this tool returns their "
        "identifiers and the output directory — it does not return file contents inline. "
        f"Refuses requests matching more than {MAX_PRODUCTS_PER_CALL} products in one call; "
        "call search_eumetsat_products first to check the count, then narrow the date range, "
        "bbox, or observations_per_day, or split into multiple calls. If start_hour is "
        "omitted, defaults to local solar noon for the request's location and reports the "
        "chosen hour in the response's `warning` field — do not assume midnight/00:00 or a "
        "full day of coverage."
    )
)
def download_eumetsat_product(args: DownloadParams) -> str:
    downloader = _new_downloader(args.product, format=args.format)
    bbox_nswe = _wsen_to_nswe(args.bbox)

    warning = None
    if args.start_hour is None:
        start_hour = _solar_noon_utc_hour(args.product, args.bbox)
        warning = (
            f"You have not specified an hour so it will be retrieved at {start_hour:02d}:00 "
            "UTC (local solar noon for this location), which maximises reflectance/"
            "illumination for VIS/NIR channels and avoids downloading every 5-15 minute "
            "repeat-cycle slot in the date range. Pass start_hour explicitly to override."
        )
    else:
        start_hour = args.start_hour

    # Preview against the actual observation windows that will be downloaded,
    # not the raw start_date..end_date span — a full day contains every 5-15
    # minute repeat-cycle slot (e.g. 96 for SEVIRI), while
    # observations_per_day/start_hour/interval_minutes narrow that down to a
    # handful of slots. Searching the raw span overcounts and can trip the
    # per-call limit even for requests that will only download 1-2 products.
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    intervals = downloader._split_time_into_daily_observations(
        start_date=start_dt,
        end_date=end_dt,
        observations_per_day=args.observations_per_day,
        start_hour=start_hour,
        interval_minutes=args.interval_minutes,
    )
    preview_count = 0
    for interval_start, interval_end in intervals:
        preview_count += downloader.search(
            start_time=interval_start.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time=interval_end.strftime("%Y-%m-%dT%H:%M:%S"),
            bounding_box=bbox_nswe,
        )["count"]

    if preview_count == 0:
        return json.dumps(
            {"product": args.product, "count": 0, "message": "No products matched this request."}
        )
    if preview_count > MAX_PRODUCTS_PER_CALL:
        raise ValueError(
            f"{preview_count} products match this request, above the "
            f"{MAX_PRODUCTS_PER_CALL}-per-call limit. Narrow the date range, bbox, or "
            "observations_per_day, or split into multiple calls."
        )

    # Detect what this call downloaded by mtime rather than deriving expected
    # filenames from each identifier: a product unpacks into exactly one
    # file for some products (SEVIRI: "<identifier>.nat") but many chunked
    # files for others (MTG FCI: ~60 "*-CHK-BODY/TRAIL-*.nc" files whose
    # names don't contain the parent identifier at all). mtime (not a
    # before/after set-diff) also correctly detects a re-download that
    # overwrites already-present files of the same name.
    import time

    download_started_at = time.time() - 1  # 1s safety margin for fs mtime resolution

    downloader.download_interval(
        start_time=args.start_date,
        end_time=args.end_date,
        bounding_box=bbox_nswe,
        method=args.method,
        observations_per_day=args.observations_per_day,
        start_hour=start_hour,
        interval_minutes=args.interval_minutes,
    )

    status_path = OUTPUT_DIR / f".{args.product}_status.json"
    downloader.initiate_download(aggregated_file=str(status_path))

    new_files = sorted(
        str(f)
        for f in (OUTPUT_DIR.glob("*") if OUTPUT_DIR.exists() else [])
        if f.is_file() and f.stat().st_mtime >= download_started_at and _is_readable_data_file(f)
    )

    downloaded = []
    downloaded_products = list(getattr(downloader, "file_list", []))
    for p in downloaded_products:
        identifier = p._browse_properties.get("identifier")
        # Best-effort per-product attribution: reliable when the identifier
        # appears in the filename (true for SEVIRI's single-file products),
        # Chunked products can also use all returned files when this call
        # downloaded just one observation. For multiple observations, do not
        # assign the full call's chunks to every product.
        matches = _downloaded_product_paths(identifier, new_files, len(downloaded_products))
        downloaded.append(
            {
                "identifier": identifier,
                "date": p._browse_properties.get("date"),
                "paths": matches,
            }
        )
    result = {
        "product": args.product,
        "method": args.method,
        "format": args.format,
        "start_hour": start_hour,
        "count": len(downloaded),
        "output_dir": str(OUTPUT_DIR),
        "products": downloaded,
        "new_files": new_files,
        "warning": warning,
    }
    if args.read and new_files:
        result["read"] = json.loads(
            _read_product(
                product=args.product,
                identifier=None,
                bands=None,
                as_geotiff=False,
                paths=new_files,
                # 'datastore' never clips server-side, so if this download
                # targeted a specific area, crop to it now that satpy has
                # loaded the (still full-disk) file — otherwise the returned
                # stats would silently cover the whole disk instead of the
                # requested AOI.
                bbox=args.bbox,
            )
        )
    return json.dumps(result, indent=2)


class ReadParams(BaseModel):
    product: str = Field(
        ..., description="Product key from list_eumetsat_products, e.g. 'MTG-FCI-L1C-FDHSI'."
    )
    paths: Optional[list[str]] = Field(
        None,
        description=(
            "Exact file paths to read, e.g. download_eumetsat_product's `new_files` "
            "or a product entry's `paths`. Takes priority over identifier. Strongly "
            "preferred for products delivered as multiple chunked files (e.g. MTG "
            "FCI, whose chunk filenames don't contain the parent identifier) — "
            "identifier-based lookup is not reliable for those."
        ),
    )
    identifier: Optional[str] = Field(
        None,
        description=(
            "Restrict to files matching this product identifier, as returned in "
            "download_eumetsat_product's `products` list. Only reliable for "
            "single-file products (e.g. SEVIRI); prefer `paths` otherwise. Omit "
            "both to read every downloaded file for this product currently under "
            "EUMETSAT_OUTPUT_DIR."
        ),
    )
    bands: Optional[list[str]] = Field(
        None,
        description=(
            "Subset of channel names to load, e.g. ['vis_06', 'ir_105']. Omit to load "
            "every band listed for this product in list_eumetsat_products."
        ),
    )
    as_geotiff: bool = Field(
        False,
        description=(
            "If true, also write each loaded band to a georeferenced GeoTIFF under "
            "EUMETSAT_OUTPUT_DIR/geotiff/ so downstream gis-mcp tools (metadata_raster, "
            "clip_raster_with_bbox, zonal_statistics, ...) can consume it."
        ),
    )
    bbox: Optional[list[float]] = Field(
        None,
        description=(
            "[west, south, east, north] in decimal degrees. Crops the scene to this "
            "AOI after satpy loads it, before computing stats or writing GeoTIFF. "
            "IMPORTANT: a 'datastore' download's bbox only filtered which full-disk "
            "products matched the search — the downloaded file itself is still the "
            "full, uncropped disk. When the original request targeted a specific "
            "city/region, pass that same bbox here to actually clip the AOI; "
            "otherwise stats/GeoTIFFs cover the whole disk. Not needed for "
            "'datatailor' downloads, which are already clipped server-side, but "
            "harmless to repeat."
        ),
    )


def _crop_stacked_band(dataset: Any, bbox: list[float]) -> Any:
    """Slice HRV area components without resampling calibrated native pixels."""
    import numpy as np
    import xarray as xr
    from pyresample.geometry import AreaDefinition, StackedAreaDefinition

    target = AreaDefinition("crop", "crop", "latlong", "EPSG:4326", 100, 100, tuple(bbox))
    pieces = []
    offset = 0
    for area in dataset.attrs["area"].defs:
        try:
            x_slice, y_slice = area.get_area_slices(target)
        except NotImplementedError as exc:
            # Pyresample uses a bare NotImplementedError for non-overlap.
            # Errors with a message indicate an unsupported geometry instead.
            if str(exc):
                raise
        else:
            if x_slice.stop > x_slice.start and y_slice.stop > y_slice.start:
                piece = dataset.isel(y=slice(offset + y_slice.start, offset + y_slice.stop), x=x_slice)
                piece.attrs = dict(dataset.attrs, area=area[y_slice, x_slice])
                pieces.append(piece)
        offset += area.height
    if not pieces:
        raise ValueError("Requested bbox does not overlap the HRV channel coverage.")
    if len(pieces) == 1:
        return pieces[0]

    # Windows can have different horizontal offsets. Pad to a common native
    # grid with NaN; do not interpolate or count padding as observed pixels.
    areas = [piece.attrs["area"] for piece in pieces]
    left = min(area.area_extent[0] for area in areas)
    right = max(area.area_extent[2] for area in areas)
    pixel_size = areas[0].pixel_size_x
    width = round((right - left) / pixel_size)
    padded = []
    padded_areas = []
    for piece, area in zip(pieces, areas):
        if not np.isclose(area.pixel_size_x, pixel_size) or area.crs != areas[0].crs:
            raise ValueError("HRV windows must share a projection and pixel spacing.")
        start = round((area.area_extent[0] - left) / pixel_size)
        piece = piece.pad(x=(start, width - start - area.width), constant_values=np.nan)
        expanded = AreaDefinition(area.area_id, area.description, area.proj_id, area.crs,
                                  width, area.height, (left, area.area_extent[1], right, area.area_extent[3]))
        x_coords, y_coords = expanded.get_proj_vectors()
        piece = piece.assign_coords(x=x_coords, y=y_coords)
        padded.append(piece)
        padded_areas.append(expanded)
    result = xr.concat(padded, dim="y", join="exact", coords="minimal", compat="override")
    result.attrs = dict(dataset.attrs, area=StackedAreaDefinition(*padded_areas).squeeze())
    return result


def _crop_selected_bands(scene: Any, bands: list[str], bbox: list[float]) -> Any:
    """Crop channels separately, accepting mixed stacked/single area types."""
    cropped_scene = scene.copy()
    for band in bands:
        if band in scene:
            dataset = scene[band]
            area = getattr(dataset, "attrs", {}).get("area")
            if hasattr(area, "defs"):
                cropped_scene[band] = _crop_stacked_band(dataset, bbox)
            else:
                cropped_scene[band] = scene.copy(datasets=[band]).crop(ll_bbox=tuple(bbox))[band]
    return cropped_scene


def _read_product(
    product: str,
    identifier: Optional[str],
    bands: Optional[list[str]],
    as_geotiff: bool,
    paths: Optional[list[str]] = None,
    bbox: Optional[list[float]] = None,
) -> str:
    """Read previously-downloaded EUMETSAT files with satpy.

    Extracts any leftover zip files, loads each requested band with the
    correct calibration — VIS/NIR channels as reflectance, IR/WV channels as
    brightness temperature, since loading both under a single calibration is
    invalid — and returns per-band summary stats, mirroring the calibration
    split in eumetsearch's `EUMDownloader._read_satpy_convert`.
    """
    if paths is not None and not paths:
        raise ValueError(
            "Explicit paths must not be empty. For a single-product chunked "
            "download, pass its nonempty `new_files` list."
        )
    if product not in products_list:
        raise ValueError(
            f"Unknown product '{product}'. Call list_eumetsat_products for valid keys."
        )

    meta = products_list[product]
    reader = meta["satpy_reader"]
    reflectance_bands = set(meta.get("reflectance_bands", []))
    bt_bands = set(meta.get("bt_bands", []))
    requested_bands = bands or meta["bands"]

    if not OUTPUT_DIR.exists():
        return json.dumps(
            {
                "product": product,
                "count": 0,
                "message": "EUMETSAT_OUTPUT_DIR does not exist yet. Call download_eumetsat_product first.",
            }
        )

    reader_candidates = [reader]
    fallback = _READER_FALLBACKS.get(reader)
    if fallback:
        reader_candidates.append(fallback)

    if not paths:
        _extract_leftover_zips(OUTPUT_DIR)

    # Always resolve via find_files_and_readers + pass its {reader: [files]}
    # dict straight to Scene(filenames=...), matching eumetsearch's own
    # working read path (ZarrExport._read_satpy_convert) exactly. Building
    # Scene(filenames=<flat list>, reader=<str>) from an explicit path list
    # instead was tried and reliably segfaulted satpy's native reader code —
    # letting satpy's own file-pattern matching select files (then narrowing
    # by `paths`/`identifier` afterwards) is the path known to be stable.
    from satpy import find_files_and_readers

    paths_set = {str(Path(p).resolve()) for p in paths} if paths else None

    found = {}
    reader_used = reader
    for candidate in reader_candidates:
        candidate_found = find_files_and_readers(
            base_dir=str(OUTPUT_DIR), reader=candidate, missing_ok=True
        )
        if paths_set:
            candidate_found = {
                r: [f for f in flist if str(Path(f).resolve()) in paths_set]
                for r, flist in candidate_found.items()
            }
        elif identifier:
            candidate_found = {
                r: [f for f in flist if identifier in str(f)]
                for r, flist in candidate_found.items()
            }
        if sum(len(flist) for flist in candidate_found.values()) > 0:
            found = candidate_found
            reader_used = candidate
            break

    files_read = sum(len(flist) for flist in found.values())
    if files_read == 0:
        return json.dumps(
            {
                "product": product,
                "count": 0,
                "message": (
                    "No downloaded files found for this product/identifier/paths. Call "
                    "download_eumetsat_product first. For chunked FCI products, pass "
                    "the exact paths returned by the download (`new_files` for a "
                    "single-product call); chunk filenames do not contain the parent identifier."
                ),
            }
        )
    reader = reader_used

    from satpy.scene import Scene

    scn = Scene(filenames=found)

    reflectance_channels = [b for b in requested_bands if b in reflectance_bands]
    bt_channels = [b for b in requested_bands if b in bt_bands]
    other_channels = [
        b for b in requested_bands if b not in reflectance_bands and b not in bt_bands
    ]

    if reflectance_channels:
        scn.load(reflectance_channels, calibration="reflectance")
    if bt_channels:
        scn.load(bt_channels, calibration="brightness_temperature")
    if other_channels:
        scn.load(other_channels)

    # 'datastore' downloads are always the full, uncropped disk — bbox during
    # download only filtered which products matched the search (see
    # download_eumetsat_product), it never clipped the file. This is the
    # actual crop step: satpy Scene.crop(ll_bbox=...) slices the loaded
    # arrays down to the AOI. Returns a new Scene rather than mutating.
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be exactly [west, south, east, north].")
        west, south, east, north = bbox
        # HRV can use a stacked area while ordinary VIS/IR bands use a single
        # area. Cropping a mixed-area Scene makes satpy compare incompatible
        # area types. Crop each requested channel independently instead.
        scn = _crop_selected_bands(scn, requested_bands, [west, south, east, north])

    import dask
    import numpy as np

    # Chunked products (e.g. MTG FCI, ~40 files per band) reliably segfault
    # under dask's default threaded scheduler — satpy/netCDF4/HDF5 reading
    # many files concurrently across worker threads isn't safely reentrant
    # in this environment. Force synchronous (single-threaded) execution for
    # every dask compute below; eumetsearch's own working read path avoids
    # this the same way (see single_thread_env in eumetsearch.utils.general,
    # though that only covers OMP/BLAS/GDAL thread counts, not dask's own
    # scheduler, which is what actually crashed here).
    with dask.config.set(scheduler="synchronous"):
        bands_out = {}
        for band in requested_bands:
            if band not in scn:
                continue
            arr = np.asarray(scn[band].values)
            bands_out[band] = {
                "shape": list(arr.shape),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }

        geotiff_paths = {}
        if as_geotiff:
            loaded = [b for b in requested_bands if b in scn]
            if loaded:
                geotiff_dir = OUTPUT_DIR / "geotiff"
                geotiff_dir.mkdir(parents=True, exist_ok=True)
                scn.save_datasets(
                    writer="geotiff",
                    datasets=loaded,
                    filename=str(geotiff_dir / ("{name}_" + product + ".tif")),
                )
                for band in loaded:
                    geotiff_paths[band] = str(geotiff_dir / f"{band}_{product}.tif")

    return json.dumps(
        {
            "product": product,
            "reader": reader,
            "files_read": files_read,
            "bands": bands_out,
            "geotiff_paths": geotiff_paths or None,
        },
        indent=2,
    )


@app.tool()
def read_eumetsat_product(args: ReadParams) -> str:
    """Read previously downloaded EUMETSAT files with satpy and return per-band summary stats.

    Call this after download_eumetsat_product. Applies the correct satpy
    calibration per channel (reflectance for VIS/NIR, brightness temperature
    for IR/WV), crops to `bbox` when given (required to actually restrict a
    'datastore' download's full-disk file to an AOI — see the `bbox` field),
    and, when as_geotiff is true, writes each band out to a georeferenced
    GeoTIFF so downstream gis-mcp tools (metadata_raster, clip_raster_with_bbox,
    zonal_statistics, ...) can consume it.
    """
    return _read_product(
        args.product, args.identifier, args.bands, args.as_geotiff, args.paths, args.bbox
    )


if __name__ == "__main__":
    app.run(transport="stdio")
