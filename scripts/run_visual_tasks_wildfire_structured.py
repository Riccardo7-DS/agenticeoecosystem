#!/usr/bin/env python
"""
run_visual_tasks_wildfire_structured.py — Run an 11-field structured wildfire
questionnaire (smoke, flame, containment, size, hotspots, intensity,
infrastructure, settlements, vegetation, weather, cloud visibility) over every
image under data/visual_tasks/, one result per image.

This is a second, richer prompt alongside run_visual_tasks_wildfire.py's
single "Do you see a wildfire?" query — same run mechanics (isolated
orchestrator session per image via get_ans_tool_path, AGENTICEO_SUBAGENTS
restricted to just the VLM agent), but asks for a fixed set of fields in a
strict `Field: value` block so answers can be parsed and scored per field
instead of only as free text.

Logs go to logs/visual_tasks_wildfire_structured_<timestamp>.log plus a JSON
summary with one entry per image (raw answer + best-effort parsed fields).
"""

import argparse
import asyncio
import json
import os
import re
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

# (field name, question, allowed options) — order defines both the question
# list and the required output block order.
FIELDS: list[tuple[str, str, list[str]]] = [
    ("Smoke", "Does this satellite image show a smoke plume from a wildfire?",
     ["Yes", "No"]),
    ("Flame", "Does this satellite image show an active flame?",
     ["Yes", "No"]),
    ("Uncontrolled", "Can you confirm that the fire is uncontrolled?",
     ["Yes", "No", "Cannot be Determined"]),
    ("Fire Size", "What is the extent of the fire in the picture in percentage?",
     ["<a percentage estimated from pixel count>", "Cannot be Determined", "No Fire Visible"]),
    ("Fire Hotspots", "Does the fire have multiple hotspots?",
     ["Multiple Hotspots", "One Hotspot", "Cannot be Determined", "No Fire Visible"]),
    ("Fire Intensity", "What is the fire intensity?",
     ["Low", "Moderate", "High", "Cannot be Determined", "No Fire Visible"]),
    ("Infrastructure", "Is there infrastructure visible close to the source of fire?",
     ["Yes - road", "Yes - city", "Yes - other", "No", "Cannot be Determined"]),
    ("Human Settlements", "Are there cities visible close to the source of fire?",
     ["Yes - big settlement", "Yes - small settlement", "No", "Cannot be Determined"]),
    ("Vegetation Type", "Describe the state of the vegetation.",
     ["Vital", "Medium Vital", "Burnt", "Missing", "Cannot be Determined"]),
    ("Weather", "Is the weather likely to sustain the fire?",
     ["Yes - high temperature and strong wind", "Maybe - high temperature",
      "Maybe - strong wind", "No", "Cannot be Determined"]),
    ("Visibility", "Are there clouds obstructing the assessment?",
     ["Yes", "No", "Cannot be Determined"]),
]

_FIELD_NAMES = [f[0] for f in FIELDS]

_QUESTION_BLOCK = "\n".join(
    f"{i}. {name} — {question} (options: {', '.join(options)})"
    for i, (name, question, options) in enumerate(FIELDS, 1)
)

_ANSWER_TEMPLATE = "\n".join(f"{name}: <answer>" for name in _FIELD_NAMES)

_QUERY_TEMPLATE = f"""Analyze the local satellite image at: {{path}}

Answer these {len(FIELDS)} questions about it, in order:

{_QUESTION_BLOCK}

Respond with EXACTLY one line per field, in this exact format, and nothing \
else before or after it — no preamble, no extra commentary, no markdown:

{_ANSWER_TEMPLATE}

Pick the single option listed for each field that best matches what the \
image shows (for "Fire Size", give a percentage instead if a fire is \
visible). If a field cannot be assessed from this image, answer "Cannot be \
Determined" for that field (or "No Fire Visible" where that option exists \
and no fire is visible at all)."""

_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(" + "|".join(re.escape(n) for n in _FIELD_NAMES) + r")(?:\*\*)?\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_fields(text: str) -> dict[str, str | None]:
    found = {}
    for m in _LINE_RE.finditer(text):
        # Normalize the matched field name back to its canonical casing.
        raw_name = m.group(1)
        canonical = next(n for n in _FIELD_NAMES if n.lower() == raw_name.lower())
        found.setdefault(canonical, m.group(2).strip())
    return {name: found.get(name) for name in _FIELD_NAMES}


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)


async def _run(m, images: list[Path], out_path: Path) -> None:
    results = []
    counts = {"parsed_ok": 0, "parse_incomplete": 0, "error": 0}

    for i, image_path in enumerate(images, 1):
        query = _QUERY_TEMPLATE.format(path=image_path)
        logger.info("[%d/%d] %s", i, len(images), image_path.name)
        try:
            raw = await m.get_ans_tool_path(query)
            parsed_response = json.loads(raw)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            results.append({"image": str(image_path), "error": str(e)})
            counts["error"] += 1
            continue

        entry = {"image": str(image_path), **parsed_response}

        if parsed_response.get("ended_in_error"):
            counts["error"] += 1
        else:
            fields = _parse_fields(parsed_response.get("final_answer", ""))
            entry["fields"] = fields
            if all(v is not None for v in fields.values()):
                counts["parsed_ok"] += 1
            else:
                counts["parse_incomplete"] += 1
                missing = [k for k, v in fields.items() if v is None]
                logger.warning("  incomplete parse, missing: %s", missing)

        logger.info("  -> %s", parsed_response.get("final_answer", "").replace("\n", " | ")[:300])
        results.append(entry)

        out_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info(
        "Done: %d parsed_ok, %d parse_incomplete, %d error (of %d total)",
        counts["parsed_ok"], counts["parse_incomplete"], counts["error"], len(images),
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

    setup_logging(log_file="visual_tasks_wildfire_structured")

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
    out_path = logs_dir / f"visual_tasks_wildfire_structured_results_{timestamp}.json"

    from agenticeoecosystem.agents import multi_agent_wildfires as m

    asyncio.run(_run(m, images, out_path))


if __name__ == "__main__":
    main()
