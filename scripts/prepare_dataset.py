import json
import random
import zipfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "images"
ANN_DIR = ROOT / "data" / "annotations"

ANN_ZIP = ANN_DIR / "annotations_trainval2017.zip"
IMAGE_ZIP = ROOT / "data" / "downloads" / "val2017.zip"

OUTPUT = ANN_DIR / "mini_pixmo.jsonl"


def read_json_from_zip(zip_path, suffix):

    with zipfile.ZipFile(zip_path) as archive:

        filename = next(
            name for name in archive.namelist()
            if name.endswith(suffix)
        )

        with archive.open(filename) as f:
            return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=100
    )

    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not ANN_ZIP.exists():
        raise FileNotFoundError(
            f"Missing annotation archive: {ANN_ZIP}"
        )

    if not IMAGE_ZIP.exists():
        raise FileNotFoundError(
            f"Missing image archive: {IMAGE_ZIP}"
        )

    print("Reading COCO annotations...")

    captions = read_json_from_zip(
        ANN_ZIP,
        "captions_val2017.json"
    )

    instances = read_json_from_zip(
        ANN_ZIP,
        "instances_val2017.json"
    )

    categories = {
        item["id"]: item["name"]
        for item in instances["categories"]
    }

    captions_by_image = {}

    for item in captions["annotations"]:

        captions_by_image.setdefault(
            item["image_id"], []
        ).append(item["caption"])

    objects_by_image = {}

    for item in instances["annotations"]:

        objects_by_image.setdefault(
            item["image_id"], []
        ).append(item)

    images = [
        item for item in captions["images"]
        if item["id"] in objects_by_image
    ]

    random.Random(42).shuffle(images)

    images = images[:args.limit]

    records = []

    print("Extracting selected images...")

    with zipfile.ZipFile(IMAGE_ZIP) as archive:

        archive_names = set(archive.namelist())

        for image_info in images:

            image_id = image_info["id"]
            file_name = image_info["file_name"]

            image_name = f"val2017/{file_name}"

            if image_name not in archive_names:
                print("Missing image:", image_name)
                continue

            destination = IMAGE_DIR / file_name

            if not destination.exists():

                with archive.open(image_name) as source:
                    with open(destination, "wb") as target:
                        target.write(source.read())

            objects = []

            width = image_info["width"]
            height = image_info["height"]

            for ann in objects_by_image[image_id]:

                x, y, w, h = ann["bbox"]

                if w <= 0 or h <= 0:
                    continue

                objects.append({

                    "label": categories[ann["category_id"]],

                    "bbox_xywh_normalized": [
                        x / width,
                        y / height,
                        w / width,
                        h / height
                    ],

                    "point_normalized": [
                        (x + w / 2) / width,
                        (y + h / 2) / height
                    ]

                })

            if not objects:
                continue

            labels = sorted(
                set(obj["label"] for obj in objects)
            )

            qa = [{
                "question": "What objects are visible in the image?",
                "answer": ", ".join(labels)
            }]

            for label in labels[:3]:

                count = sum(
                    obj["label"] == label
                    for obj in objects
                )

                qa.append({

                    "question": f"How many {label} are visible?",

                    "answer": str(count)

                })

            caption_list = captions_by_image.get(
                image_id, []
            )

            records.append({

                "image_id": image_id,

                "image": f"data/images/{file_name}",

                "caption": (
                    caption_list[0]
                    if caption_list else ""
                ),

                "qa": qa,

                "objects": objects

            })

    with open(OUTPUT, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\nDataset preparation complete!")

    print("Images:", len(records))

    print("Annotations:", OUTPUT)


if __name__ == "__main__":
    main()