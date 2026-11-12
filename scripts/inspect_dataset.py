import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "data" / "annotations" / "mini_pixmo.jsonl"

records = []

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("\n========== MINI-PIXMO DATASET REPORT ==========\n")

print("Total images:", len(records))

caption_count = sum(
    bool(record["caption"]) for record in records
)

qa_count = sum(
    len(record["qa"]) for record in records
)

object_count = sum(
    len(record["objects"]) for record in records
)

print("Images with captions:", caption_count)
print("Question-answer pairs:", qa_count)
print("Object annotations:", object_count)

labels = Counter(
    obj["label"]
    for record in records
    for obj in record["objects"]
)

print("\n========== TOP OBJECT CATEGORIES ==========\n")

for label, count in labels.most_common(15):
    print(f"{label:20s}: {count}")

missing_images = []
invalid_points = []

for record in records:

    image_path = ROOT / record["image"]

    if not image_path.exists():
        missing_images.append(record["image"])

    for obj in record["objects"]:

        x, y = obj["point_normalized"]

        if not (0 <= x <= 1 and 0 <= y <= 1):
            invalid_points.append(record["image_id"])

print("\n========== DATA QUALITY ==========\n")

print("Missing images:", len(missing_images))
print("Invalid coordinates:", len(invalid_points))

if missing_images:
    print("Missing image examples:", missing_images[:5])

if invalid_points:
    print("Invalid point examples:", invalid_points[:5])

print("\n========== FINAL STATUS ==========\n")

if not missing_images and not invalid_points:
    print("SUCCESS: Dataset validation passed!")
else:
    print("WARNING: Please investigate the dataset issues.")