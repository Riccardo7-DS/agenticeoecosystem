#!/usr/bin/env python
"""
run_visual_decision_wildfire.py — Single-shot test of the wildfire
orchestrator's TASKING ORDER reasoning: run multi_agent_coordination on one
random image from the FireImages dataset, attach a synthetic acquisition
time/location to it, and ask the orchestrator to combine its
capability-based DATA ACQUISITION PLAN with real satellite overpass
predictions (fetched live from the TLE sub-agent) into a ranked tasking
order.

No image in this dataset carries real acquisition metadata (confirmed: no
per-scene timestamp exists anywhere in the CDSE/EUMETSAT/VLM plumbing), so
this script assigns a RANDOM acquisition timestamp (kept close to "now" so
TLE propagation stays meaningful — TLEs degrade in accuracy days/weeks past
their epoch) and a random lat/lon as a stand-in for real scene metadata.

Restricts sub-agent loading to vlm + tle (AGENTICEO_SUBAGENTS=vlm,tle)
before importing the orchestrator, so no other sub-agent's MCP toolset has
to spin up. Logs go to logs/visual_decision_wildfire_<timestamp>.log (via
the project's setup_logging) plus a JSON result file with the same
timestamp.
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("AGENTICEO_SUBAGENTS", "vlm,tle")

from dotenv import load_dotenv
load_dotenv()

import logging
from agenticeoecosystem.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_QUERY_TEMPLATE = (
    "A wildfire-monitoring image is attached below for analysis. No real "
    "acquisition metadata exists for this scene, so assume it was acquired "
    "at {timestamp_utc} UTC at approximately latitude {lat:.4f}, longitude "
    "{lon:.4f}.\n\n"
    "Image: {path}\n\n"
    "1. Analyze the image for wildfire evidence and produce your FEATURES "
    "ANALYSIS and DATA ACQUISITION PLAN as usual, identifying which "
    "additional satellites would help close any evidence gaps.\n"
    "2. Call the TLE sub-agent's `get_acquisition_schedule` tool ONCE with "
    "the full list of non-geostationary satellites you recommended (its "
    "`satellites` argument), latitude {lat:.4f}, longitude {lon:.4f}, "
    "start_utc={timestamp_utc}, and window_hours=24 — this returns every "
    "one of their predicted overpasses in a single time-sorted list, so "
    "prefer it over calling get_target_overpasses satellite-by-satellite. "
    "For any geostationary platform you recommend instead (e.g. MSG/SEVIRI, "
    "MTG/FCI), call get_satellite_position for its current position "
    "instead, since it has no meaningful overpass. You MUST actually make "
    "these tool calls now and read their results before writing the "
    "TASKING ORDER below — never write 'I will now fetch...' or any other "
    "statement of future intent as a substitute for calling the tool.\n"
    "3. Combine the capability-based plan with the actual timing you just "
    "fetched into a TASKING ORDER: a ranked, numbered sequence of which "
    "satellite to task first, second, etc. For each entry, state its next "
    "realistic acquisition time from the tool results (not a guess) and "
    "the capability/urgency trade-off that justifies its position (e.g. "
    "'tasked first despite lower burned-area capability because it is "
    "overhead in 40 minutes, versus the higher-capability option that is "
    "6 hours away')."
)


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)


def _pick_image(images_dir: Path, rng: random.Random) -> Path:
    images = _find_images(images_dir)
    if not images:
        raise SystemExit(f"No images found under {images_dir}")
    return rng.choice(images)


def _random_acquisition_context(rng: random.Random) -> tuple[str, float, float]:
    """Synthesize a plausible (timestamp_utc, lat, lon) for a test image
    that carries no real acquisition metadata. The timestamp is kept close
    to "now" (+/- 12h) so the tasking-order step's TLE propagation stays
    meaningful."""
    now = datetime.now(timezone.utc)
    offset = timedelta(hours=rng.uniform(-12, 12))
    timestamp_utc = (now + offset).strftime("%Y-%m-%dT%H:%M:%SZ")
    lat = rng.uniform(-60.0, 70.0)
    lon = rng.uniform(-180.0, 180.0)
    return timestamp_utc, lat, lon


async def _run(
    m,
    image_path: Path,
    timestamp_utc: str,
    lat: float,
    lon: float,
    out_path: Path,
) -> None:
    query = _QUERY_TEMPLATE.format(
        path=image_path, timestamp_utc=timestamp_utc, lat=lat, lon=lon
    )
    logger.info("Image: %s", image_path)
    logger.info(
        "Assumed acquisition: %s UTC at (%.4f, %.4f)", timestamp_utc, lat, lon
    )

    result = {
        "image": str(image_path),
        "assumed_acquisition_utc": timestamp_utc,
        "assumed_lat": lat,
        "assumed_lon": lon,
    }
    try:
        raw = await m.get_ans_tool_path(query)
        result.update(json.loads(raw))
    except Exception as e:
        logger.error("FAILED: %s", e)
        result["error"] = str(e)

    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Result written to %s", out_path)

    final_answer = result.get("final_answer")
    if final_answer:
        logger.info("\n%s\n%s\n%s", "=" * 80, final_answer, "=" * 80)
    if result.get("tools_called"):
        logger.info(
            "Tools called: %s",
            [t["function_name"] for t in result["tools_called"]],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        default=str(_ROOT / "data" / "visual_tasks" / "images"),
        help="Directory to search recursively for .jpg/.jpeg/.png images "
             "(the FireImages dataset). Ignored if --image is given.",
    )
    parser.add_argument(
        "--image", default=None,
        help="Explicit image path to use instead of picking one at random.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for image selection and the synthetic "
             "acquisition time/location, for reproducible test runs.",
    )
    args = parser.parse_args()

    setup_logging(log_file="visual_decision_wildfire")

    rng = random.Random(args.seed)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"--image path does not exist: {image_path}")
    else:
        image_path = _pick_image(Path(args.images_dir), rng)

    timestamp_utc, lat, lon = _random_acquisition_context(rng)

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"visual_decision_wildfire_result_{out_timestamp}.json"

    # Imported here (synchronous context, after AGENTICEO_SUBAGENTS is set and
    # logging is configured) because the module itself calls asyncio.run() at
    # import time to set up its session — nesting that inside our own running
    # event loop would raise.
    from agenticeoecosystem.agents import multi_agent_coordination as m

    asyncio.run(_run(m, image_path, timestamp_utc, lat, lon, out_path))


if __name__ == "__main__":
    main()
