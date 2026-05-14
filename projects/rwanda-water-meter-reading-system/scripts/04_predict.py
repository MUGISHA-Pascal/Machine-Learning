#!/usr/bin/env python3
"""
04_predict.py

Run YOLO inference on test images and save visualized predictions.

Fully automatic:
- Finds latest best.pt from training_runs/
- Uses dataset/images/test
- Saves annotated images
- Saves prediction .txt files
- Saves confidence scores

Usage:
    python3 scripts/04_predict.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


TRAINING_ROOT = Path("runs/detect/training_runs")
TEST_IMAGES_DIR = Path("dataset/images/test")
OUTPUT_PROJECT = Path("prediction_outputs")
OUTPUT_NAME = "test_predictions"

IMAGE_SIZE = 640
CONFIDENCE = 0.25

SAVE_TXT = True
SAVE_CONF = True

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))

    if not best_models:
        raise FileNotFoundError(
            f"No best.pt model found inside: {training_root}"
        )

    return max(best_models, key=lambda p: p.stat().st_mtime)


def iter_test_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def main() -> None:
    model_path = find_latest_best_model(TRAINING_ROOT)

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {TEST_IMAGES_DIR}"
        )

    test_images = list(iter_test_images(TEST_IMAGES_DIR))

    if not test_images:
        raise RuntimeError(f"No supported test images found in: {TEST_IMAGES_DIR}")

    OUTPUT_PROJECT.mkdir(parents=True, exist_ok=True)

    print("========== YOLO TEST PREDICTION ==========")
    print(f"Model:       {model_path}")
    print(f"Test folder: {TEST_IMAGES_DIR}")
    print(f"Images:      {len(test_images)}")
    print(f"Output root: {OUTPUT_PROJECT}")
    print(f"Run name:    {OUTPUT_NAME}")
    print(f"Image size:  {IMAGE_SIZE}")
    print(f"Confidence:  {CONFIDENCE}")
    print()

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(TEST_IMAGES_DIR),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        save_txt=SAVE_TXT,
        save_conf=SAVE_CONF,
        show_labels=True,
        show_conf=False,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=False,
    )

    output_dir = OUTPUT_PROJECT / OUTPUT_NAME
    label_dir = output_dir / "labels"

    print("\nPrediction completed. [√]")
    print(f"Annotated images saved in: {output_dir}")
    print(f"Prediction txt files saved in: {label_dir}")

    print("\nPer-image summary:")
    for image_path, result in zip(test_images, results):
        num_boxes = 0 if result.boxes is None else len(result.boxes)
        print(f"  - {image_path.name}: {num_boxes} detection(s)")


if __name__ == "__main__":
    main()
