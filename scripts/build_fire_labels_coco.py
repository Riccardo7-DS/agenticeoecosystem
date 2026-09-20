"""Convert labels.xlsx smoke/burn-area columns into a COCO-format classification JSON
for the FireImages-Jul-Aug image set.

Usage: uv run --with openpyxl --with pillow python scripts/build_fire_labels_coco.py
(run from the agenticeoecosystem/ directory)
"""

import json
import re
from pathlib import Path

import openpyxl
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LABELS_XLSX = ROOT / "data/visual_tasks/labels/labels.xlsx"
IMAGES_DIR = ROOT / "data/visual_tasks/images/FireImages-Jul-Aug"
OUTPUT_JSON = ROOT / "data/visual_tasks/labels/FireImages-Jul-Aug.json"

CATEGORY_SMOKE = 1
CATEGORY_BURN_AREA = 2
CATEGORY_SMOKE_AND_BURN_AREA = 3


def normalize(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return value


def load_id_to_labels():
    wb = openpyxl.load_workbook(LABELS_XLSX, data_only=True)
    ws = wb["Foglio1"]
    rows = list(ws.iter_rows(values_only=True))

    id_to_labels = {}
    for row in rows[1:]:
        if row[0] == "Lat":  # phantom repeated header row mid-sheet
            continue
        if row[4] is None:
            continue
        try:
            image_id = int(str(row[4]).strip())
        except ValueError:
            continue
        smoke = normalize(row[5])
        burn_area = normalize(row[6])
        id_to_labels[image_id] = (smoke, burn_area)
    return id_to_labels


def main():
    id_to_labels = load_id_to_labels()

    files = sorted(IMAGES_DIR.iterdir())
    images = []
    annotations = []
    ann_id = 1
    skipped_unmatched = []
    skipped_maybe = []

    for f in files:
        m = re.match(r"0*(\d+)\.jpg$", f.name)
        if not m:
            continue
        image_id = int(m.group(1))

        if image_id not in id_to_labels:
            skipped_unmatched.append(image_id)
            continue

        smoke, burn_area = id_to_labels[image_id]
        if "maybe" in (smoke, burn_area):
            skipped_maybe.append(image_id)
            continue

        with Image.open(f) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": f.name,
                "width": width,
                "height": height,
            }
        )

        if smoke == "yes" and burn_area == "yes":
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_SMOKE_AND_BURN_AREA,
                }
            )
            ann_id += 1
        elif smoke == "yes":
            annotations.append(
                {"id": ann_id, "image_id": image_id, "category_id": CATEGORY_SMOKE}
            )
            ann_id += 1
        elif burn_area == "yes":
            annotations.append(
                {"id": ann_id, "image_id": image_id, "category_id": CATEGORY_BURN_AREA}
            )
            ann_id += 1

    coco = {
        "info": {
            "description": "FireImages-Jul-Aug smoke/burn-area classification labels",
            "date_created": "2026-09-03",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_SMOKE, "name": "smoke", "supercategory": "wildfire"},
            {"id": CATEGORY_BURN_AREA, "name": "burn_area", "supercategory": "wildfire"},
            {
                "id": CATEGORY_SMOKE_AND_BURN_AREA,
                "name": "smoke_and_burn_area",
                "supercategory": "wildfire",
            },
        ],
    }

    OUTPUT_JSON.write_text(json.dumps(coco, indent=2))

    print(f"Total files on disk: {len(files)}")
    print(f"Unmatched (no spreadsheet row), skipped: {len(skipped_unmatched)} -> {skipped_unmatched}")
    print(f"Skipped for 'maybe': {len(skipped_maybe)} -> {skipped_maybe}")
    print(f"Images written: {len(images)}")
    print(f"Annotations written: {len(annotations)}")
    smoke_count = sum(1 for a in annotations if a["category_id"] == CATEGORY_SMOKE)
    burn_count = sum(1 for a in annotations if a["category_id"] == CATEGORY_BURN_AREA)
    both_count = sum(
        1 for a in annotations if a["category_id"] == CATEGORY_SMOKE_AND_BURN_AREA
    )
    print(f"  smoke only: {smoke_count}, burn_area only: {burn_count}, smoke_and_burn_area: {both_count}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
