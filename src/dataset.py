import json
from pathlib import Path
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset


class MiniPixMoDataset(Dataset):

    def __init__(
        self,
        root,
        annotation_file,
        processor,
        max_samples=None
    ):

        self.root = Path(root)
        self.processor = processor

        with open(annotation_file, encoding="utf-8") as f:
            records = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        if max_samples is not None:
            records = records[:max_samples]

        self.samples = []

        self.total_objects = 0
        self.unique_localization_targets = 0
        self.skipped_ambiguous_targets = 0
        self.invalid_boxes = 0

        for rec in records:

            objects = rec.get("objects", [])

            label_counts = Counter(
                obj["label"].lower().strip()
                for obj in objects
            )

            self.total_objects += len(objects)

            # ---------------------------------
            # 1. Caption supervision
            # ---------------------------------

            if rec.get("caption"):

                self.samples.append({
                    "record": rec,
                    "prompt": "Describe the image:",
                    "answer": rec["caption"],
                    "box": [0.0, 0.0, 1.0, 1.0],
                    "box_valid": False
                })

            # ---------------------------------
            # 2. General QA supervision
            # ---------------------------------

            for qa in rec.get("qa", []):

                self.samples.append({
                    "record": rec,
                    "prompt": qa["question"],
                    "answer": qa["answer"],
                    "box": [0.0, 0.0, 1.0, 1.0],
                    "box_valid": False
                })

            # ---------------------------------
            # 3. Unique-object bounding boxes
            # ---------------------------------

            for obj in objects:

                label = obj["label"].lower().strip()

                # Only use categories appearing once
                if label_counts[label] != 1:
                    self.skipped_ambiguous_targets += 1
                    continue

                bbox = obj.get("bbox_xywh_normalized")

                if not bbox or len(bbox) != 4:
                    self.invalid_boxes += 1
                    continue

                try:
                    x, y, width, height = map(float, bbox)
                except (TypeError, ValueError):
                    self.invalid_boxes += 1
                    continue

                # Reject invalid or non-positive dimensions
                if width <= 0 or height <= 0:
                    self.invalid_boxes += 1
                    continue

                # Convert COCO xywh into xyxy
                x_min = max(0.0, min(1.0, x))
                y_min = max(0.0, min(1.0, y))
                x_max = max(0.0, min(1.0, x + width))
                y_max = max(0.0, min(1.0, y + height))

                # Ensure a valid box remains after clipping
                if x_max <= x_min or y_max <= y_min:
                    self.invalid_boxes += 1
                    continue

                box = [
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ]

                self.samples.append({
                    "record": rec,
                    "prompt": f"Where is the {label}?",
                    "answer": f"The {label} is visible in the image.",
                    "box": box,
                    "box_valid": True
                })

                self.unique_localization_targets += 1

        # ---------------------------------
        # Dataset statistics
        # ---------------------------------

        self.localization_samples = sum(
            1 for sample in self.samples
            if sample["box_valid"]
        )

        print("\nMiniPixMo Experiment 9 Dataset Summary")
        print("---------------------------------------")
        print("Images:", len(records))
        print("Total object instances:", self.total_objects)
        print(
            "Unique bounding-box targets:",
            self.unique_localization_targets
        )
        print(
            "Skipped ambiguous targets:",
            self.skipped_ambiguous_targets
        )
        print("Invalid bounding boxes:", self.invalid_boxes)
        print("Total training samples:", len(self.samples))
        print(
            "Bounding-box training samples:",
            self.localization_samples
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        record = sample["record"]

        image_path = self.root / record["image"]

        image = Image.open(image_path).convert("RGB")

        pixels = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"][0]

        return {
            "pixel_values": pixels,

            "prompt": sample["prompt"],

            "answer": sample["answer"],

            "box": torch.tensor(
                sample["box"],
                dtype=torch.float32
            ),

            "box_valid": torch.tensor(
                sample["box_valid"],
                dtype=torch.bool
            )
        }
