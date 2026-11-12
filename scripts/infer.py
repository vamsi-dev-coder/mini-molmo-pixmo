import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from src.image_processing import get_image_processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MiniMolmo


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--question",
        default="What objects are visible?"
    )
    parser.add_argument(
        "--task",
        choices=["caption", "qa"],
        default="qa"
    )

    args = parser.parse_args()

    # Image path
    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Processor and model
    processor = get_image_processor()

    model = MiniMolmo().to(device)

    # Load trained checkpoint
    checkpoint = ROOT / "checkpoints" / "mini_molmo.pt"

    if not checkpoint.exists():
        raise FileNotFoundError("Training checkpoint not found.")

    state = torch.load(checkpoint, map_location="cpu")

    model.connector.load_state_dict(state["connector"])
    model.point_head.load_state_dict(state["point_head"])

    model.eval()

    # Process image
    image = Image.open(image_path).convert("RGB")

    pixels = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(device)

    # Select prompt
    if args.task == "caption":
        prompt = "Describe the image:"
    else:
        prompt = args.question

    # Generate prediction
    with torch.inference_mode():
        text, point = model.generate(pixels, prompt)

    print("\nGenerated text:", text)
    print("Normalized point (x,y):", point)

    pixel_x = round(point[0] * image.width)
    pixel_y = round(point[1] * image.height)

    print("Pixel coordinate:", (pixel_x, pixel_y))


if __name__ == "__main__":
    main()