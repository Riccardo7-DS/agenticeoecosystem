""" WORK IN PROGRESS SCRIPT: Compare VLM wildfire-detection outputs against the FireImages-Jul-Aug ground-truth
COCO labels, write a per-image comparison CSV, and print accuracy metrics.

The two log files each ran the same orchestrator pipeline but with a different VLM
sub-agent backend (confirmed in the companion .log files):
  - visual_tasks_wildfire_results_20260903_091640.json -> vlm_agent model "Qwen3-VL"
  - visual_tasks_wildfire_results_20260903_140127.json -> vlm_agent model "Gemma4-12B"

Each model was asked a free-text question about wildfire presence, not asked to
output a structured label directly, and answer style/length varies across runs (short
templated "Yes/No, ..." openers in some runs, long structured bullet-point answers in
others). To map free text onto the 4-class ground-truth taxonomy
(none / smoke / burn_area / smoke_and_burn_area) reliably across all of these styles,
this script uses an LLM classifier (the locally-served vLLM endpoint from
agents/subagents/vlm_init.py) that reads the *entire* answer and returns one of the
four labels — this replaced an earlier regex/keyword heuristic that broke whenever
the answer's phrasing didn't exactly match one of a fixed set of templated openers.
Classifications are cached to disk (by hash of the answer text) so re-running the
script doesn't re-classify unchanged answers.

Usage (from agenticeoecosystem/): uv run python scripts/eval_wildfire_labels.py
"""

import csv
import hashlib
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from agenticeoecosystem.agents.subagents.vlm_init import VLM_BASE_URL

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GT_JSON = ROOT / "data/visual_tasks/labels/FireImages-Jul-Aug.json"
LOGS = {
    "Qwen3-VL": ROOT / "logs/visual_tasks_wildfire_results_20260903_091640.json",
    "Gemma4-12B": ROOT / "logs/visual_tasks_wildfire_results_20260903_140127.json",
}
OUTPUT_CSV = ROOT / "output/wildfire_vlm_label_comparison.csv"
CLASSIFIER_CACHE = ROOT / "output/.wildfire_classify_cache.json"

CATEGORY_NAMES = {0: "none", 1: "smoke", 2: "burn_area", 3: "smoke_and_burn_area"}
_NAME_TO_LABEL = {v: k for k, v in CATEGORY_NAMES.items()}

# Same OpenAI-compatible endpoint the VLM sub-agent talks to (see vlm_init.py) — this
# is a plain text classification call, no image, so any chat-capable model served
# there works. Model id must match the currently-served name (check with
# `curl http://localhost:18001/v1/models`).
_CLASSIFIER_MODEL = "Qwen3-VL"
_client = OpenAI(base_url=VLM_BASE_URL, api_key="EMPTY")

_CLASSIFY_PROMPT = """You are grading a vision-language model's free-text answer about \
whether a satellite image shows a wildfire. Classify the ANSWER below into exactly \
one of these four categories, based on everything it says (read the full answer, not \
just the opening sentence — it may be long and structured with multiple points):

- none: no wildfire evidence at all — no active flame/smoke and no burn scar.
- smoke: active smoke/plume/haze described, but no scorched/burned/blackened ground.
- burn_area: scorched/blackened/charred/ash-covered ground (a burn scar) described, \
but no active smoke/plume/haze.
- smoke_and_burn_area: BOTH active smoke/plume AND a burn scar are described.

Active flame/glow alone (with no smoke or burn scar mentioned) also counts as "smoke" \
— it indicates active fire without a described burn scar.

<ANSWER>
{answer}
</ANSWER>

Respond with exactly this format and nothing else:
<CLASSIFICATION>none|smoke|burn_area|smoke_and_burn_area</CLASSIFICATION>"""

_TAG_RE = re.compile(
    r"<CLASSIFICATION>\s*(none|smoke|burn_area|smoke_and_burn_area)\s*</CLASSIFICATION>",
    re.IGNORECASE,
)


def _classify_call(text: str) -> int:
    for attempt in range(2):
        resp = _client.chat.completions.create(
            model=_CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=40,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(answer=text)}],
        )
        out = resp.choices[0].message.content or ""
        m = _TAG_RE.search(out)
        if m:
            return _NAME_TO_LABEL[m.group(1).lower()]
        logger.warning("Classifier attempt %d: unparseable output %r for answer %r", attempt, out[:200], text[:200])
    raise ValueError(f"Classifier could not resolve a verdict from: {text!r}")


def _load_cache() -> dict:
    if CLASSIFIER_CACHE.exists():
        return json.loads(CLASSIFIER_CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CLASSIFIER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_CACHE.write_text(json.dumps(cache, indent=2))


def classify_all(texts: list[str]) -> dict[str, int]:
    """Classify a batch of answer texts in parallel, using a disk cache keyed by
    a hash of the text so unchanged answers aren't re-classified on rerun."""
    cache = _load_cache()
    keys = {t: hashlib.sha256(t.encode()).hexdigest() for t in texts}
    todo = [t for t in texts if keys[t] not in cache]

    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(tqdm(pool.map(_classify_call, todo), total=len(todo), desc="Classifying answers"))
        for t, label in zip(todo, results):
            cache[keys[t]] = label
        _save_cache(cache)

    return {t: cache[keys[t]] for t in texts}


def load_ground_truth():
    gt = json.loads(GT_JSON.read_text())
    id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    id_to_label = {img_id: 0 for img_id in id_to_file}
    for ann in gt["annotations"]:
        id_to_label[ann["image_id"]] = ann["category_id"]
    return id_to_file, id_to_label


def load_predictions(path, labels_by_text):
    entries = json.loads(path.read_text())
    preds = {}
    for e in entries:
        m = re.match(r"0*(\d+)\.jpg$", Path(e["image"]).name)
        if not m:
            continue  # e.g. tile_XX.png, not part of FireImages-Jul-Aug
        image_id = int(m.group(1))
        preds[image_id] = {"label": labels_by_text[e["final_answer"]], "raw": e["final_answer"]}
    return preds


def main():
    id_to_file, id_to_gt = load_ground_truth()

    all_texts = []
    for path in LOGS.values():
        entries = json.loads(path.read_text())
        all_texts.extend(e["final_answer"] for e in entries)
    labels_by_text = classify_all(all_texts)

    model_preds = {name: load_predictions(path, labels_by_text) for name, path in LOGS.items()}

    model_names = list(LOGS.keys())
    rows = []
    correct_counts = {name: 0 for name in model_names}
    binary_correct_counts = {name: 0 for name in model_names}
    confusion = {name: Counter() for name in model_names}  # (gt, pred) -> count

    for image_id in sorted(id_to_file):
        gt_label = id_to_gt[image_id]
        row = {
            "image_id": image_id,
            "file_name": id_to_file[image_id],
            "ground_truth": CATEGORY_NAMES[gt_label],
        }
        for name in model_names:
            entry = model_preds[name].get(image_id)
            if entry is None:
                row[f"{name}_prediction"] = "MISSING"
                row[f"{name}_correct"] = ""
                row[f"{name}_raw_answer"] = ""
                continue
            pred_label = entry["label"]
            is_correct = pred_label == gt_label
            row[f"{name}_prediction"] = CATEGORY_NAMES[pred_label]
            row[f"{name}_correct"] = is_correct
            row[f"{name}_raw_answer"] = entry["raw"]

            correct_counts[name] += int(is_correct)
            binary_correct_counts[name] += int((pred_label > 0) == (gt_label > 0))
            confusion[name][(gt_label, pred_label)] += 1
        rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "file_name", "ground_truth"]
    for name in model_names:
        fieldnames += [f"{name}_prediction", f"{name}_correct", f"{name}_raw_answer"]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"Evaluated {n} images against ground truth.")
    print(f"Wrote comparison CSV: {OUTPUT_CSV}\n")

    for name in model_names:
        acc = correct_counts[name] / n
        bin_acc = binary_correct_counts[name] / n
        print(f"=== {name} ===")
        print(f"  4-class accuracy (exact category match): {correct_counts[name]}/{n} = {acc:.1%}")
        print(f"  binary accuracy (any-positive vs none):   {binary_correct_counts[name]}/{n} = {bin_acc:.1%}")
        print("  confusion matrix (rows=ground truth, cols=prediction):")
        labels = [0, 1, 2, 3]
        header = "        " + "".join(f"{CATEGORY_NAMES[c]:>22}" for c in labels)
        print(header)
        for gt_l in labels:
            row_counts = [confusion[name][(gt_l, pred_l)] for pred_l in labels]
            print(f"  {CATEGORY_NAMES[gt_l]:>6}" + "".join(f"{c:>22}" for c in row_counts))
        print()


if __name__ == "__main__":
    main()
