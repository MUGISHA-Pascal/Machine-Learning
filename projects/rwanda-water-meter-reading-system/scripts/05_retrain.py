#!/usr/bin/env python3
"""
05_retrain.py

Retrain / continue training YOLO using the latest best.pt model found in training_runs/.

Fully automatic:
- Finds latest training_runs/*/weights/best.pt
- Uses dataset/data.yaml
- Creates a new training run
- Can be run again and again after many training runs

Usage:
    python3 scripts/05_retrain.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


TRAINING_ROOT = Path("training_runs")
DATA_YAML = Path("dataset/data.yaml")

EPOCHS = 10
IMAGE_SIZE = 640
BATCH_SIZE = 8

PROJECT_NAME = "training_runs"
RUN_PREFIX = "water_meter_retrain"


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))

    if not best_models:
        raise FileNotFoundError(
            f"No best.pt model found inside: {training_root}"
        )

    return max(best_models, key=lambda p: p.stat().st_mtime)


def make_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    latest_best = find_latest_best_model(TRAINING_ROOT)
    run_name = make_run_name(RUN_PREFIX)

    print("========== YOLO RETRAINING ==========")
    print(f"Starting from: {latest_best}")
    print(f"Data YAML:     {DATA_YAML}")
    print(f"New run name:  {run_name}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Image size:    {IMAGE_SIZE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print()

    model = YOLO(str(latest_best))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=run_name,
    )

    print("\nRetraining completed. [√]")
    print(f"New run saved under: {PROJECT_NAME}/{run_name}")


if __name__ == "__main__":
    main()
