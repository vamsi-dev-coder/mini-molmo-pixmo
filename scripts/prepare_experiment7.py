import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ANN_ZIP = ROOT / "data/annotations/annotations_trainval2017.zip"
IMAGE_ZIP = ROOT / "data/downloads/train2017.zip"

IMAGE_DIR = ROOT / "data/images/train2017"

OUTPUT_DIR = ROOT / "data/annotations/experiment7"

ADDITIONAL_IMAGES = 100
SEED = 42


def read_json_from_zip(zip_path, suffix):

    with zipfile.ZipFile(zip_path) as archive:

        filename = next(
            name for name in archive.namelist()
            if name.endswith(suffix)
        )

        with archive.open(filename) as f:
            return json.load(f)


def main():

    if not ANN_ZIP.exists():
        raise FileNotFoundError(ANN_ZIP)

    if not IMAGE_ZIP.exists():
        raise FileNotFoundError(IMAGE_ZIP)

    print("Reading COCO training annotations...")

    captions = read_json_from_zip(
        ANN_ZIP,
        "captions_train2017.json"
    )

    instances = read_json_from_zip(
        ANN_ZIP,
        "instances_train2017.json"
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

    random.Random(SEED).shuffle(images)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing_train = ROOT / "data/annotations/train.jsonl"
    existing_val = ROOT / "data/annotations/val.jsonl"

    with open(existing_train, encoding="utf-8") as f:
        train_records = [
            json.loads(line)
            for line in f if line.strip()
        ]

    with open(existing_val, encoding="utf-8") as f:
        val_records = [
            json.loads(line)
            for line in f if line.strip()
        ]

    existing_ids = {
        record["image_id"]
        for record in train_records + val_records
        if "image_id" in record
    }

    selected = []

    for image in images:

        if image["id"] in existing_ids:
            continue

        if len(selected) >= ADDITIONAL_IMAGES:
            break

        selected.append(image)

    if len(selected) < ADDITIONAL_IMAGES:
        raise RuntimeError("Not enough additional images found.")

    print("Selected images:", len(selected))
    print("Extracting selected images only...")

    new_records = []

    with zipfile.ZipFile(IMAGE_ZIP) as archive:

        archive_names = set(archive.namelist())

        for index, image_info in enumerate(selected):

            image_id = image_info["id"]
            filename = image_info["file_name"]

            archive_name = f"train2017/{filename}"

            if archive_name not in archive_names:
                continue

            destination = IMAGE_DIR / filename

            if not destination.exists():

                with archive.open(archive_name) as source:
                    with open(destination, "wb") as target:
                        target.write(source.read())

            width = image_info["width"]
            height = image_info["height"]

            objects = []

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

            labels = sorted({
                obj["label"] for obj in objects
            })

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

            new_records.append({
                "image_id": image_id,
                "image": f"data/images/train2017/{filename}",
                "caption": caption_list[0] if caption_list else "",
                "qa": qa,
                "objects": objects
            })

            print(
                f"Processed {index + 1}/{len(selected)}"
            )

    combined_train = train_records + new_records

    train_output = OUTPUT_DIR / "train.jsonl"
    val_output = OUTPUT_DIR / "val.jsonl"

    with open(train_output, "w", encoding="utf-8") as f:
        for record in combined_train:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(val_output, "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nExperiment 7 dataset complete!")
    print("Original training images:", len(train_records))
    print("Additional training images:", len(new_records))
    print("Total training images:", len(combined_train))
    print("Validation images:", len(val_records))
    print("Training annotations:", train_output)
    print("Validation annotations:", val_output)


if __name__ == "__main__":
    main()