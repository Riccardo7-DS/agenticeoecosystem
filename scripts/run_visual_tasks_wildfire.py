#!/usr/bin/env python
"""
run_visual_tasks_wildfire.py — Run the wildfire-detection VLM query over every
image under data/visual_tasks/, logging one result per image.

Restricts sub-agent loading to just the VLM agent (AGENTICEO_SUBAGENTS=vlm)
before importing the orchestrator, so no other sub-agent's MCP toolset has to
spin up. Logs go to logs/visual_tasks_wildfire_<timestamp>.log (via the
project's setup_logging) plus a JSON summary with one entry per image.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("AGENTICEO_SUBAGENTS", "vlm")

from dotenv import load_dotenv
load_dotenv()

import logging
from agenticeoecosystem.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_QUERY_TEMPLATE = "Do you see a wildfire in the local image? {path}"


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)


async def _run(m, images: list[Path], out_path: Path) -> None:
    results = []
    counts = {"wildfire": 0, "no_wildfire": 0, "error": 0}

    for i, image_path in enumerate(images, 1):
        query = _QUERY_TEMPLATE.format(path=image_path)
        logger.info("[%d/%d] %s", i, len(images), image_path.name)
        try:
            raw = await m.get_ans_tool_path(query)
            parsed = json.loads(raw)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            results.append({"image": str(image_path), "error": str(e)})
            counts["error"] += 1
            continue

        if parsed.get("ended_in_error"):
            counts["error"] += 1
        elif "yes" in parsed["final_answer"].strip().lower()[:10]:
            counts["wildfire"] += 1
        else:
            counts["no_wildfire"] += 1

        logger.info("  -> %s", parsed["final_answer"].replace("\n", " ")[:200])
        results.append({"image": str(image_path), **parsed})

        out_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info(
        "Done: %d wildfire, %d no_wildfire, %d error (of %d total)",
        counts["wildfire"], counts["no_wildfire"], counts["error"], len(images),
    )
    logger.info("Results written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        default=str(_ROOT / "data" / "visual_tasks"),
        help="Directory to search recursively for .jpg/.jpeg/.png images.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images (for a quick test run).",
    )
    args = parser.parse_args()

    setup_logging(log_file="visual_tasks_wildfire")

    images = _find_images(Path(args.images_dir))
    if args.limit:
        images = images[: args.limit]
    if not images:
        logger.warning("No images found under %s", args.images_dir)
        return
    logger.info("Found %d image(s) under %s", len(images), args.images_dir)

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"visual_tasks_wildfire_results_{timestamp}.json"

    # Imported here (synchronous context, after AGENTICEO_SUBAGENTS is set and
    # logging is configured) because the module itself calls asyncio.run() at
    # import time to set up its session — nesting that inside our own running
    # event loop would raise.
    from agenticeoecosystem.agents import multi_agent_wildfires as m

    asyncio.run(_run(m, images, out_path))


if __name__ == "__main__":
    main()
