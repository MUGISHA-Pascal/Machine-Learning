#!/usr/bin/env python3
"""
02_prepare_dataset.py

Prepare a complete YOLO dataset from raw student-labeled folders.

Input:
    extracted_labeled_data/
        student folders...
        images + matching .txt YOLO labels

Output:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── reports/
    └── data.yaml

Usage:
    python3 02_prepare_dataset.py raw_dataset dataset

Optional:
    python3 02_prepare_dataset.py raw_dataset dataset --train 0.7 --val 0.2 --test 0.1
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

IGNORE_DIR_NAMES = {
    "__MACOSX",
    "venv",
    ".venv",
    "env",
    ".env",
    "Lib",
    "Include",
    "Scripts",
    "site-packages",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}

IGNORE_FILE_NAMES = {
    "desktop.ini",
    ".ds_store",
    "label_images.py",
    "README.md",
}

CLASS_NAMES = [
    "meter",
    "window",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "u",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare clean YOLO train/val/test dataset from raw labeled data."
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Raw folder containing student-labeled data",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output YOLO dataset folder",
    )

    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def should_ignore_path(path: Path) -> bool:
    parts_lower = {part.lower() for part in path.parts}

    for ignored in IGNORE_DIR_NAMES:
        if ignored.lower() in parts_lower:
            return True

    if path.name.lower() in {name.lower() for name in IGNORE_FILE_NAMES}:
        return True

    return False


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_files(source_dir: Path):
    images_by_stem = {}
    labels_by_stem = {}

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue

        if should_ignore_path(path):
            continue

        if is_image(path):
            images_by_stem.setdefault(path.stem, []).append(path)

        elif path.suffix.lower() == ".txt":
            labels_by_stem.setdefault(path.stem, []).append(path)

    return images_by_stem, labels_by_stem


def choose_file(paths):
    return sorted(paths, key=lambda p: str(p).lower())[0]


def validate_label_line(
    line: str,
    label_file: Path,
    line_number: int,
    num_classes: int,
) -> List[str]:
    errors = []
    parts = line.strip().split()

    if len(parts) != 5:
        errors.append(f"{label_file} line {line_number}: expected 5 values, got {len(parts)}")
        return errors

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        errors.append(f"{label_file} line {line_number}: contains non-numeric values")
        return errors

    if not (0 <= class_id < num_classes):
        errors.append(f"{label_file} line {line_number}: invalid class_id {class_id}")

    for name, value in [
        ("x_center", x_center),
        ("y_center", y_center),
        ("width", width),
        ("height", height),
    ]:
        if not (0.0 <= value <= 1.0):
            errors.append(f"{label_file} line {line_number}: {name}={value} outside [0,1]")

    if width <= 0:
        errors.append(f"{label_file} line {line_number}: width must be > 0")

    if height <= 0:
        errors.append(f"{label_file} line {line_number}: height must be > 0")

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        errors.append(
            f"{label_file} line {line_number}: box extends outside image "
            f"(x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f})"
        )

    return errors


def validate_pair(image_path: Path, label_path: Path) -> List[str]:
    errors = []

    try:
        img = cv2.imread(str(image_path))
    except cv2.error as e:
        errors.append(f"OpenCV failed to read image, possibly too large: {image_path}")
        errors.append(f"OpenCV error: {e}")
        return errors

    if img is None:
        errors.append(f"Unreadable image: {image_path}")
        return errors

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        errors.append(f"Invalid image dimensions: {image_path} -> {w}x{h}")

    if label_path.stat().st_size == 0:
        errors.append(f"Empty label file: {label_path}")
        return errors

    with label_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    for line_number, line in enumerate(lines, start=1):
        errors.extend(
            validate_label_line(
                line=line,
                label_file=label_path,
                line_number=line_number,
                num_classes=len(CLASS_NAMES),
            )
        )

    return errors


def prepare_pairs(source_dir: Path, reports_dir: Path):
    images_by_stem, labels_by_stem = collect_files(source_dir)

    image_stems = set(images_by_stem.keys())
    label_stems = set(labels_by_stem.keys())

    matched_stems = sorted(image_stems & label_stems)
    missing_label_stems = sorted(image_stems - label_stems)
    orphan_label_stems = sorted(label_stems - image_stems)

    valid_pairs: List[Tuple[Path, Path, str]] = []

    mapping_lines = []
    missing_label_lines = []
    orphan_label_lines = []
    duplicate_lines = []
    validation_error_lines = []

    for stem in missing_label_stems:
        missing_label_lines.append(
            f"MISSING_LABEL: {stem}\n"
            + "".join(f"  image: {p}\n" for p in images_by_stem[stem])
        )

    for stem in orphan_label_stems:
        orphan_label_lines.append(
            f"ORPHAN_LABEL: {stem}\n"
            + "".join(f"  label: {p}\n" for p in labels_by_stem[stem])
        )

    clean_index = 1

    for stem in matched_stems:
        image_path = choose_file(images_by_stem[stem])
        label_path = choose_file(labels_by_stem[stem])

        if len(images_by_stem[stem]) > 1 or len(labels_by_stem[stem]) > 1:
            duplicate_lines.append(
                f"DUPLICATE_STEM: {stem}\n"
                f"  images:\n"
                + "".join(f"    - {p}\n" for p in images_by_stem[stem])
                + "  labels:\n"
                + "".join(f"    - {p}\n" for p in labels_by_stem[stem])
            )

        errors = validate_pair(image_path, label_path)

        if errors:
            validation_error_lines.append(
                f"INVALID_PAIR_SKIPPED: {stem}\n"
                f"  image: {image_path}\n"
                f"  label: {label_path}\n"
                + "".join(f"  error: {err}\n" for err in errors)
            )
            continue

        new_stem = f"wm_clean_{clean_index:06d}"
        valid_pairs.append((image_path, label_path, new_stem))

        mapping_lines.append(
            f"{new_stem}\n"
            f"  image_from: {image_path}\n"
            f"  label_from: {label_path}\n"
        )

        clean_index += 1

    (reports_dir / "mapping.txt").write_text("\n".join(mapping_lines), encoding="utf-8")
    (reports_dir / "missing_labels.txt").write_text("\n".join(missing_label_lines), encoding="utf-8")
    (reports_dir / "orphan_labels.txt").write_text("\n".join(orphan_label_lines), encoding="utf-8")
    (reports_dir / "duplicates.txt").write_text("\n".join(duplicate_lines), encoding="utf-8")
    (reports_dir / "validation_errors.txt").write_text("\n".join(validation_error_lines), encoding="utf-8")

    return valid_pairs, images_by_stem, labels_by_stem, validation_error_lines


def copy_split_pairs(
    pairs: List[Tuple[Path, Path, str]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    for split_name, split_pairs in splits.items():
        for image_path, label_path, new_stem in split_pairs:
            new_image_name = new_stem + image_path.suffix.lower()
            new_label_name = new_stem + ".txt"

            shutil.copy2(image_path, images_dir / split_name / new_image_name)
            shutil.copy2(label_path, labels_dir / split_name / new_label_name)

    return {
        "total": total,
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }


def create_data_yaml(output_dir: Path):
    data_yaml = output_dir / "data.yaml"

    dataset_path = output_dir.resolve()

    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]

    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Train, val, and test ratios must add up to 1.0")

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    clean_output_dir(output_dir)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    valid_pairs, images_by_stem, labels_by_stem, validation_error_lines = prepare_pairs(
        source_dir=source_dir,
        reports_dir=reports_dir,
    )

    if not valid_pairs:
        print("No valid image-label pairs found. Dataset was not created.")
        return

    split_counts = copy_split_pairs(
        pairs=valid_pairs,
        output_dir=output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    create_data_yaml(output_dir)

    print("\n========== YOLO DATASET PREPARATION COMPLETE ==========")
    print(f"Source folder: {source_dir}")
    print(f"Output folder: {output_dir}")
    print()
    print(f"Images found: {sum(len(v) for v in images_by_stem.values())}")
    print(f"Labels found: {sum(len(v) for v in labels_by_stem.values())}")
    print(f"Valid image-label pairs used: {split_counts['total']}")
    print(f"Invalid pairs skipped: {len(validation_error_lines)}")
    print()
    print("Split:")
    print(f"Train: {split_counts['train']}")
    print(f"Val:   {split_counts['val']}")
    print(f"Test:  {split_counts['test']}")
    print()
    print("Created:")
    print(f"  {output_dir / 'images' / 'train'}")
    print(f"  {output_dir / 'images' / 'val'}")
    print(f"  {output_dir / 'images' / 'test'}")
    print(f"  {output_dir / 'labels' / 'train'}")
    print(f"  {output_dir / 'labels' / 'val'}")
    print(f"  {output_dir / 'labels' / 'test'}")
    print(f"  {output_dir / 'data.yaml'}")
    print(f"  {reports_dir}")
    print()
    print("√ READY FOR TRAINING")


if __name__ == "__main__":
    main()
