import sys
import json
import argparse

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import MiniPixMoDataset
from src.image_processing import get_image_processor
from src.model import MiniMolmo


def collate(batch):
    return {
        "pixel_values": torch.stack(
            [x["pixel_values"] for x in batch]
        ),
        "prompts": [x["prompt"] for x in batch],
        "answers": [x["answer"] for x in batch],
        "boxes": torch.stack(
            [x["box"] for x in batch]
        ),
        "box_valid": torch.stack(
            [x["box_valid"] for x in batch]
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)

    args = parser.parse_args()

    # ---------------------------------
    # Device
    # ---------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------
    # Dataset
    # ---------------------------------

    processor = get_image_processor()

    train_file = (
        ROOT / "data" / "annotations"
        / "experiment7" / "train.jsonl"
    )

    dataset = MiniPixMoDataset(
        root=ROOT,
        annotation_file=train_file,
        processor=processor,
        max_samples=args.max_samples
    )

    # ---------------------------------
    # Weighted sampling
    # ---------------------------------

    weights = [
        3.0 if sample["box_valid"] else 1.0
        for sample in dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = MiniMolmo().to(device)

    # ---------------------------------
    # Load Experiment 7 checkpoint
    # ---------------------------------

    initial_checkpoint = (
        ROOT / "checkpoints"
        / "experiment7_expanded_data.pt"
    )

    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            f"Experiment 7 checkpoint not found: {initial_checkpoint}"
        )

    state = torch.load(
        initial_checkpoint,
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

    # Do NOT load point_head.
    # Experiment 9 uses a newly initialized bbox_head.

    print("\nInitialized compatible layers from:")
    print(initial_checkpoint)

    print("Bounding-box head: freshly initialized")

    # ---------------------------------
    # Trainable parameters
    # ---------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    total_params = sum(
        p.numel() for p in trainable_params
    )

    print("\nTraining examples:", len(dataset))
    print(
        "Bounding-box examples:",
        dataset.localization_samples
    )
    print("Trainable parameters:", total_params)

    # ---------------------------------
    # Optimizer
    # ---------------------------------

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # ---------------------------------
    # Output paths
    # ---------------------------------

    checkpoint_dir = ROOT / "checkpoints"
    results_dir = ROOT / "results"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir / "experiment9_bbox_localization.pt"
    )

    history_path = (
        results_dir / "experiment9_training_history.json"
    )

    history = []

    # ---------------------------------
    # Training
    # ---------------------------------

    for epoch in range(args.epochs):

        model.train()

        total_loss_sum = 0.0
        text_loss_sum = 0.0
        bbox_loss_sum = 0.0

        progress = tqdm(
            loader,
            desc=f"Experiment 9 - Epoch {epoch + 1}/{args.epochs}"
        )

        for batch in progress:

            pixels = batch["pixel_values"].to(device)
            boxes = batch["boxes"].to(device)
            box_valid = batch["box_valid"].to(device)

            prompts = batch["prompts"]
            answers = batch["answers"]

            optimizer.zero_grad(set_to_none=True)

            total_loss, text_loss, bbox_loss = model.training_loss(
                pixels,
                prompts,
                answers,
                boxes,
                box_valid
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0
            )

            optimizer.step()

            total_loss_sum += total_loss.item()
            text_loss_sum += text_loss.item()
            bbox_loss_sum += bbox_loss.item()

            progress.set_postfix(
                total=f"{total_loss.item():.4f}",
                text=f"{text_loss.item():.4f}",
                bbox=f"{bbox_loss.item():.4f}"
            )

        # ---------------------------------
        # Epoch summary
        # ---------------------------------

        steps = len(loader)

        epoch_result = {
            "epoch": epoch + 1,
            "total_loss": total_loss_sum / steps,
            "text_loss": text_loss_sum / steps,
            "bbox_loss": bbox_loss_sum / steps
        }

        history.append(epoch_result)

        print("\nEpoch results:")
        print(json.dumps(epoch_result, indent=2))

        # ---------------------------------
        # Save checkpoint after each epoch
        # ---------------------------------

        torch.save(
            {
                "connector": model.connector.state_dict(),
                "question_to_vision":
                    model.question_to_vision.state_dict(),
                "visual_attention":
                    model.visual_attention.state_dict(),
                "bbox_head":
                    model.bbox_head.state_dict(),
                "experiment": 9,
                "epoch": epoch + 1,
                "training_history": history
            },
            checkpoint_path
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    print("\nExperiment 9 training completed.")
    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()
