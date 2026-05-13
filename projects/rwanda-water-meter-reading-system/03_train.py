#!/usr/bin/env python3
"""
03_train.py

Train a YOLO model using Ultralytics.

Install first:
    pip install ultralytics

Usage:
    python3 scripts/03_train.py

Or:
    python3 scripts/03_train.py --data dataset/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8 --name water_meter_corrected_names
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for water meter detection and reading."
    )

    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", default="training_runs", help="Output project folder")
    parser.add_argument("--name", default="water_meter_yolov8n", help="Experiment name")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
