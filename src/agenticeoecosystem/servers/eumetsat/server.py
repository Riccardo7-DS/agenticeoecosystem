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
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - EUMETSAT-MCP - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from eumetsearch import EUMDownloader, products_list  # noqa: E402

app = FastMCP("eumetsat-mcp")

OUTPUT_DIR = Path(
    os.environ.get("EUMETSAT_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "downloads"))
)

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

    Args:
        product: product key from list_eumetsat_products, e.g. "MTG-FCI-L1C-FDHSI".
        start_date, end_date: ISO datetime strings 'YYYY-MM-DDTHH:MM:SS' (UTC).
        bbox: optional [west, south, east, north] bounding box in decimal
              degrees. Omit for the full disk.

    Returns:
        JSON with "count" and a list of matching {"identifier", "date"} entries.
    """
    downloader = _new_downloader(product)
    result = downloader.search(
        start_time=start_date, end_time=end_date, bounding_box=_wsen_to_nswe(bbox)
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
    start_hour: int = Field(10, description="UTC hour of the first observation each day.")
    interval_minutes: int = Field(10, description="Length of each observation slot in minutes.")


@app.tool(
    description=(
        "Download EUMETSAT MTG FCI / MSG SEVIRI product files for a time window. "
        "Files are saved to disk under EUMETSAT_OUTPUT_DIR and this tool returns their "
        "identifiers and the output directory — it does not return file contents inline. "
        f"Refuses requests matching more than {MAX_PRODUCTS_PER_CALL} products in one call; "
        "call search_eumetsat_products first to check the count, then narrow the date range, "
        "bbox, or observations_per_day, or split into multiple calls."
    )
)
def download_eumetsat_product(args: DownloadParams) -> str:
    downloader = _new_downloader(args.product, format=args.format)
    bbox_nswe = _wsen_to_nswe(args.bbox)

    preview = downloader.search(
        start_time=args.start_date, end_time=args.end_date, bounding_box=bbox_nswe
    )
    if preview["count"] == 0:
        return json.dumps(
            {"product": args.product, "count": 0, "message": "No products matched this request."}
        )
    if preview["count"] > MAX_PRODUCTS_PER_CALL:
        raise ValueError(
            f"{preview['count']} products match this request, above the "
            f"{MAX_PRODUCTS_PER_CALL}-per-call limit. Narrow the date range, bbox, or "
            "observations_per_day, or split into multiple calls."
        )

    downloader.download_interval(
        start_time=args.start_date,
        end_time=args.end_date,
        bounding_box=bbox_nswe,
        method=args.method,
        observations_per_day=args.observations_per_day,
        start_hour=args.start_hour,
        interval_minutes=args.interval_minutes,
    )

    status_path = OUTPUT_DIR / f".{args.product}_status.json"
    downloader.initiate_download(aggregated_file=str(status_path))

    downloaded = [
        {
            "identifier": p._browse_properties.get("identifier"),
            "date": p._browse_properties.get("date"),
        }
        for p in getattr(downloader, "file_list", [])
    ]
    return json.dumps(
        {
            "product": args.product,
            "method": args.method,
            "format": args.format,
            "count": len(downloaded),
            "output_dir": str(OUTPUT_DIR),
            "products": downloaded,
        },
        indent=2,
    )


if __name__ == "__main__":
    app.run(transport="stdio")
