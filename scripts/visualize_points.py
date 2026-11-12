import sys
import json
import argparse

from pathlib import Path
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_processing import get_image_processor
from src.model import MiniMolmo


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


def convert_bbox(bbox, width, height):

    x, y, w, h = map(float, bbox)

    return [
        int(x * width),
        int(y * height),
        int((x + w) * width),
        int((y + h) * height)
    ]


def calculate_iou(predicted, ground_truth):

    px1, py1, px2, py2 = predicted
    gx1, gy1, gx2, gy2 = ground_truth

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    intersection = (
        max(0, ix2 - ix1) *
        max(0, iy2 - iy1)
    )

    pred_area = (
        max(0, px2 - px1) *
        max(0, py2 - py1)
    )

    gt_area = (
        max(0, gx2 - gx1) *
        max(0, gy2 - gy1)
    )

    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0.0


def draw_visualization(
    image,
    predicted_box,
    ground_truth_box,
    label,
    iou,
    output_path
):

    draw = ImageDraw.Draw(image)

    # Ground truth: blue
    draw.rectangle(
        ground_truth_box,
        outline="blue",
        width=4
    )

    # Prediction: red
    draw.rectangle(
        predicted_box,
        outline="red",
        width=4
    )

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (10, 10),
        f"Object: {label}",
        fill="yellow",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    draw.text(
        (10, 35),
        f"IoU: {iou:.4f}",
        fill="yellow",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    draw.text(
        (10, 60),
        "Blue: Ground Truth | Red: Prediction",
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    image.save(output_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/experiment9_bbox_localization.pt"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=20
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    # ---------------------------------
    # Load validation records
    # ---------------------------------

    val_file = ROOT / "data" / "annotations" / "val.jsonl"

    with open(val_file, encoding="utf-8") as f:
        all_records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    records = []

    for record in all_records:

        if select_unique_object(record) is not None:
            records.append(record)

        if len(records) >= args.max_samples:
            break

    print("Validation images:", len(records))

    # ---------------------------------
    # Load model
    # ---------------------------------

    processor = get_image_processor()

    model = MiniMolmo().to(device)

    checkpoint_path = ROOT / args.checkpoint

    state = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.connector.load_state_dict(state["connector"])
    model.question_to_vision.load_state_dict(
        state["question_to_vision"]
    )
    model.visual_attention.load_state_dict(
        state["visual_attention"]
    )
    model.bbox_head.load_state_dict(state["bbox_head"])

    model.eval()

    # ---------------------------------
    # Output directory
    # ---------------------------------

    output_dir = ROOT / "results" / "experiment9_visualizations"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    iou_scores = []

    # ---------------------------------
    # Generate visualizations
    # ---------------------------------

    with torch.inference_mode():

        for index, record in enumerate(records):

            obj = select_unique_object(record)

            label = obj["label"].lower().strip()

            image_path = ROOT / record["image"]

            image = Image.open(
                image_path
            ).convert("RGB")

            width, height = image.size

            pixels = processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(device)

            patches = model.visual_features(pixels)

            predicted_normalized = model.predict_boxes(
                patches,
                [f"Where is the {label}?"]
            )[0].cpu().tolist()

            # Convert normalized predictions to image coordinates
            predicted_box = [
                int(predicted_normalized[0] * width),
                int(predicted_normalized[1] * height),
                int(predicted_normalized[2] * width),
                int(predicted_normalized[3] * height)
            ]

            ground_truth_box = convert_bbox(
                obj["bbox_xywh_normalized"],
                width,
                height
            )

            iou = calculate_iou(
                predicted_box,
                ground_truth_box
            )

            iou_scores.append(iou)

            output_path = (
                output_dir / f"image_{index + 1:02d}.jpg"
            )

            draw_visualization(
                image,
                predicted_box,
                ground_truth_box,
                label,
                iou,
                output_path
            )

            print(
                f"Image {index + 1}/{len(records)} | "
                f"{label} | IoU: {iou:.4f} | "
                f"Saved: {output_path.name}"
            )

    mean_iou = sum(iou_scores) / len(iou_scores)

    print("\nVisualization completed.")
    print("Images generated:", len(iou_scores))
    print("Mean IoU:", round(mean_iou, 4))
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()
