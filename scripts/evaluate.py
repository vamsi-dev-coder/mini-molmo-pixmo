import sys
import json
import re
import argparse

from pathlib import Path
from collections import Counter

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_processing import get_image_processor
from src.model import MiniMolmo


def token_f1(prediction, reference):

    prediction_tokens = re.findall(
        r"\b\w+\b",
        prediction.lower()
    )

    reference_tokens = re.findall(
        r"\b\w+\b",
        reference.lower()
    )

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = set(prediction_tokens) & set(reference_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def select_unique_object(record):

    objects = record.get("objects", [])

    counts = Counter(
        obj["label"].lower().strip()
        for obj in objects
    )

    for obj in objects:

        label = obj["label"].lower().strip()

        if counts[label] != 1:
            continue

        bbox = obj.get("bbox_xywh_normalized")

        if bbox and len(bbox) == 4:
            return obj

    return None


def convert_bbox_xywh_to_xyxy(bbox):

    x, y, width, height = map(float, bbox)

    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + width))
    y2 = max(0.0, min(1.0, y + height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ground-truth bounding box")

    return [x1, y1, x2, y2]


def calculate_iou(predicted, ground_truth):

    px1, py1, px2, py2 = predicted
    gx1, gy1, gx2, gy2 = ground_truth

    # Intersection
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)

    intersection = (
        intersection_width * intersection_height
    )

    # Areas
    predicted_area = (
        max(0.0, px2 - px1)
        * max(0.0, py2 - py1)
    )

    ground_truth_area = (
        max(0.0, gx2 - gx1)
        * max(0.0, gy2 - gy1)
    )

    union = predicted_area + ground_truth_area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def coordinate_error(predicted, ground_truth):

    return sum(
        abs(p - g)
        for p, g in zip(predicted, ground_truth)
    ) / 4.0


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/experiment9_bbox_localization.pt"
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=9
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=20
    )

    args = parser.parse_args()

    # --------------------------------
    # Device
    # --------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------
    # Load validation records
    # --------------------------------

    val_file = (
        ROOT / "data" / "annotations" / "val.jsonl"
    )

    if not val_file.exists():
        raise FileNotFoundError(val_file)

    with open(val_file, encoding="utf-8") as f:

        all_records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    records = []

    for record in all_records:

        obj = select_unique_object(record)

        if obj is not None:
            records.append(record)

        if len(records) >= args.max_samples:
            break

    if not records:
        raise ValueError(
            "No validation records with unique object targets."
        )

    print("Validation images:", len(records))

    # --------------------------------
    # Load processor and model
    # --------------------------------

    processor = get_image_processor()

    model = MiniMolmo().to(device)

    checkpoint_path = ROOT / args.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    state = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.connector.load_state_dict(
        state["connector"]
    )

    model.question_to_vision.load_state_dict(
        state["question_to_vision"]
    )

    model.visual_attention.load_state_dict(
        state["visual_attention"]
    )

    model.bbox_head.load_state_dict(
        state["bbox_head"]
    )

    model.eval()

    # --------------------------------
    # Evaluation
    # --------------------------------

    caption_scores = []
    iou_scores = []
    coordinate_errors = []

    with torch.inference_mode():

        for i, record in enumerate(records):

            image_path = ROOT / record["image"]

            image = Image.open(
                image_path
            ).convert("RGB")

            pixels = processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(device)

            # --------------------------------
            # Caption evaluation
            # --------------------------------

            generated_text, predicted_box = model.generate(
                pixels,
                "Describe the image:"
            )

            caption_score = token_f1(
                generated_text,
                record["caption"]
            )

            caption_scores.append(caption_score)

            # --------------------------------
            # Bounding-box evaluation
            # --------------------------------

            obj = select_unique_object(record)

            label = obj["label"].lower().strip()

            prompt = f"Where is the {label}?"

            patches = model.visual_features(pixels)

            predicted_box = model.predict_boxes(
                patches,
                [prompt]
            )[0].cpu().tolist()

            ground_truth_box = convert_bbox_xywh_to_xyxy(
                obj["bbox_xywh_normalized"]
            )

            # Calculate metrics
            iou = calculate_iou(
                predicted_box,
                ground_truth_box
            )

            coord_error = coordinate_error(
                predicted_box,
                ground_truth_box
            )

            iou_scores.append(iou)
            coordinate_errors.append(coord_error)

            print(
                f"Image {i + 1}/{len(records)} | "
                f"Object: {label} | "
                f"Caption F1: {caption_score:.4f} | "
                f"IoU: {iou:.4f} | "
                f"Box error: {coord_error:.4f}"
            )

    # --------------------------------
    # Calculate aggregate metrics
    # --------------------------------

    mean_caption_f1 = sum(caption_scores) / len(caption_scores)

    mean_iou = sum(iou_scores) / len(iou_scores)

    mean_coordinate_error = (
        sum(coordinate_errors) / len(coordinate_errors)
    )

    iou_50_accuracy = sum(
        score >= 0.50 for score in iou_scores
    ) / len(iou_scores)

    iou_75_accuracy = sum(
        score >= 0.75 for score in iou_scores
    ) / len(iou_scores)

    metrics = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "num_images": len(records),
        "caption_token_f1_mean": mean_caption_f1,
        "mean_iou": mean_iou,
        "iou_50_accuracy": iou_50_accuracy,
        "iou_75_accuracy": iou_75_accuracy,
        "mean_normalized_bbox_coordinate_error": mean_coordinate_error,
        "individual_iou_scores": iou_scores
    }

    # --------------------------------
    # Save results
    # --------------------------------

    results_dir = ROOT / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        results_dir
        / f"experiment{args.experiment}_evaluation_metrics.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print("\nExperiment 9 Evaluation Results:")
    print(json.dumps(metrics, indent=2))

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
