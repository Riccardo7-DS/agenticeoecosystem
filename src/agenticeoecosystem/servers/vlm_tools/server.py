"""
MCP server for the VLM sub-agent.

Exposes two simple, composable tools:

1. resolve_images(path, limit=10)
   - Cheap discovery: given a path (single file or directory), returns the
     list of image files that WOULD be loaded, without reading any bytes.
   - Lets the orchestrator/agent sanity-check "did I point at the right
     thing" before paying the cost of loading + sending image bytes.

2. load_images(path, limit=10)
   - Actually reads up to `limit` images and returns them as MCP
     ImageContent blocks (base64-encoded), plus a TextContent summary.
   - This is the tool result that ends up back in the model's context as
     multimodal content for the VLM to reason over.

Design choices:
- Directory mode sorts filenames for determinism and takes the first `limit`.
- Hard cap of 10 images regardless of what the caller passes, to protect
  against context blowup on a big vLLM VLM call.
- Only common raster image extensions are considered "images".
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from PIL import Image
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm-mcp-server")

mcp = FastMCP("vlm-tools")

HARD_LIMIT = 10  # never exceed this, no matter what the caller asks for

_DISPLAY_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
_IMAGE_EXTENSIONS = _DISPLAY_EXTENSIONS | _GEOTIFF_EXTENSIONS

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def _geotiff_to_png_bytes(path: Path) -> bytes:
    """Convert a (possibly multi-band, non-8-bit) GeoTIFF to a percentile-
    stretched 8-bit PNG in memory.

    Raw scientific rasters (reflectance, radiance, temperature, etc.) aren't
    directly viewable — this picks the first 1-3 bands and auto-stretches
    each to its 2nd-98th percentile range, the same approach used for GEE
    quicklooks, so the VLM gets a sensibly-contrasted image regardless of
    the source dataset's native value range.
    """
    with rasterio.open(path) as src:
        count = min(src.count, 3)
        bands = src.read(list(range(1, count + 1))).astype("float64")

    stretched = np.zeros_like(bands, dtype=np.uint8)
    for i in range(bands.shape[0]):
        band = bands[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1
        stretched[i] = np.clip((np.clip(band, lo, hi) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    if stretched.shape[0] == 1:
        img = Image.fromarray(stretched[0], mode="L")
    else:
        img = Image.fromarray(np.moveaxis(stretched, 0, -1), mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _discover(path: str, limit: int) -> List[Path]:
    """Resolve a path into a sorted, capped list of image file paths.

    Raises FileNotFoundError / ValueError with a clear message on bad input,
    so the tool can surface a useful error to the agent instead of a raw
    traceback.
    """
    p = Path(path).expanduser()

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    capped_limit = min(limit, HARD_LIMIT)

    if p.is_file():
        if not _is_image(p):
            raise ValueError(
                f"'{p.name}' is not a recognized image type "
                f"(supported: {sorted(_IMAGE_EXTENSIONS)})"
            )
        return [p]

    if p.is_dir():
        candidates = sorted(
            (f for f in p.iterdir() if f.is_file() and _is_image(f)),
            key=lambda f: f.name,
        )
        if not candidates:
            raise ValueError(f"No image files found in directory: {p}")
        return candidates[:capped_limit]

    raise ValueError(f"Path is neither a file nor a directory: {p}")


@mcp.tool()
def resolve_images(path: str, limit: int = HARD_LIMIT) -> str:
    """Discover which image(s) a path resolves to, without loading them.

    Use this first to confirm what will be sent to the VLM before calling
    load_images, especially for directories.

    Args:
        path: Absolute or relative path to a single image file, or a
            directory containing images.
        limit: Max number of images to consider if `path` is a directory.
            Capped at 10 regardless of the value passed.

    Returns:
        A human-readable summary listing the resolved image paths, the
        total count of image files actually present (if a directory), and
        whether the result was truncated by the limit.
    """
    try:
        all_in_dir = None
        p = Path(path).expanduser()
        if p.is_dir():
            all_in_dir = sorted(
                f.name for f in p.iterdir() if f.is_file() and _is_image(f)
            )

        resolved = _discover(path, limit)
        lines = [f"Resolved {len(resolved)} image(s) from: {path}"]
        for f in resolved:
            lines.append(f"  - {f}")

        if all_in_dir is not None and len(all_in_dir) > len(resolved):
            lines.append(
                f"Note: directory contains {len(all_in_dir)} image(s) total; "
                f"only the first {len(resolved)} will be loaded "
                f"(hard limit = {HARD_LIMIT})."
            )
        return "\n".join(lines)

    except (FileNotFoundError, ValueError) as e:
        logger.warning("resolve_images failed for %s: %s", path, e)
        return f"ERROR: {e}"


@mcp.tool()
def load_images(path: str, limit: int = HARD_LIMIT) -> list:
    """Load image(s) at a path and return them as image content for the VLM.

    Given a single image file, loads just that image. Given a directory,
    loads up to `limit` images (sorted by filename), capped at 10.

    Args:
        path: Absolute or relative path to a single image file, or a
            directory containing images.
        limit: Max number of images to load if `path` is a directory.
            Capped at 10 regardless of the value passed.

    Returns:
        A list of content blocks: one TextContent summary followed by one
        ImageContent block per loaded image, ready to be reasoned over.
    """
    try:
        resolved = _discover(path, limit)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("load_images failed for %s: %s", path, e)
        return [TextContent(type="text", text=f"ERROR: {e}")]

    blocks = [
        TextContent(
            type="text",
            text=f"Loaded {len(resolved)} image(s) from '{path}':\n"
            + "\n".join(f"  - {f.name}" for f in resolved),
        )
    ]

    for f in resolved:
        if f.suffix.lower() in _GEOTIFF_EXTENSIONS:
            try:
                data = _geotiff_to_png_bytes(f)
            except Exception as e:
                logger.warning("Failed to convert GeoTIFF %s: %s", f, e)
                blocks.append(TextContent(type="text", text=f"ERROR converting {f.name} to a viewable image: {e}"))
                continue
            mime_type = "image/png"
        else:
            try:
                data = f.read_bytes()
            except OSError as e:
                logger.warning("Failed to read %s: %s", f, e)
                blocks.append(TextContent(type="text", text=f"ERROR reading {f.name}: {e}"))
                continue
            mime_type = _MIME_TYPES.get(f.suffix.lower(), "application/octet-stream")

        b64 = base64.b64encode(data).decode("utf-8")
        blocks.append(ImageContent(type="image", data=b64, mimeType=mime_type))

    return blocks


if __name__ == "__main__":
    mcp.run(transport="stdio")