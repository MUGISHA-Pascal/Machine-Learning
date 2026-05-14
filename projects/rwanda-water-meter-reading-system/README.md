# Rwanda Water Meter Reading System

An AI-based computer vision project for automatic water meter detection and reading using YOLO object detection.

This project includes:

- dataset collection
- image labeling
- dataset preparation
- YOLO training
- retraining
- prediction
- reading reconstruction

The system is trained to detect:

- water meter
- reading window
- digits `0–9`
- unclear digits (`unknown`)

---

# Features

The project provides tools for:

- collecting water meter datasets
- labeling water meter images
- rotating and cleaning images
- validating YOLO datasets
- preparing train/val/test datasets
- training YOLO models
- retraining from latest best model
- predicting on test images
- reconstructing final meter readings

---

# Project Structure

```text
rwanda-water-meter-reading-system/
├── raw_dataset/
│   ├── images/
│   └── labels/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── reports/
│   └── data.yaml
│
├── scripts/
│   ├── 01_label_images.py
│   ├── 02_prepare_yolo_dataset.py
│   ├── 03_train_yolo.py
│   ├── 04_predict_on_test_images.py
│   └── 05_retrain.py
│
├── training_runs/
├── prediction_outputs/
└── README.md
```

---

# Supported Classes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | meter | Full physical water meter |
| 1 | window | Reading/display window |
| 2 | 0 | Digit 0 |
| 3 | 1 | Digit 1 |
| 4 | 2 | Digit 2 |
| 5 | 3 | Digit 3 |
| 6 | 4 | Digit 4 |
| 7 | 5 | Digit 5 |
| 8 | 6 | Digit 6 |
| 9 | 7 | Digit 7 |
| 10 | 8 | Digit 8 |
| 11 | 9 | Digit 9 |
| 12 | unknown | Unclear/unreadable digit |

---

# Installation

## Install dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install PySide6
```

---

# Dataset Structure

Expected raw dataset structure:

```text
raw_dataset/
├── images/
└── labels/
```

Example:

```text
raw_dataset/
├── images/
│   ├── wm_0001.jpg
│   ├── wm_0002.jpg
│   └── ...
│
└── labels/
    ├── wm_0001.txt
    ├── wm_0002.txt
    └── ...
```

---

# YOLO Label Format

Each line inside a `.txt` file follows:

```text
class_id x_center y_center width height
```

Example:

```text
2 0.309783 0.450836 0.028986 0.030898
```

Meaning:

```text
class 2 → digit 0
```

Coordinates are normalized between `0` and `1`.

---

# Workflow

```text
Data Collection
 ↓
Data Labeling
 ↓
Dataset Preparation
 ↓
YOLO Training
 ↓
Retraining
 ↓
Prediction
 ↓
Reading Reconstruction
```

---

# How to Label Images

Run:

```bash
python3 scripts/01_label_images.py
```

Features:

- draw bounding boxes
- resize boxes
- move boxes
- rotate images
- remove bad images
- save YOLO labels

---

# Prepare YOLO Dataset

Run:

```bash
python3 scripts/02_prepare_yolo_dataset.py raw_dataset dataset
```

The script automatically:

- validates labels
- removes invalid pairs
- creates train/val/test splits
- generates `data.yaml`

---

# Train YOLO

Run:

```bash
python3 scripts/03_train_yolo.py
```

Training results are saved in:

```text
training_runs/
```

---

# Retrain from Latest Best Model

Run:

```bash
python3 scripts/05_retrain.py
```

The script automatically finds the latest:

```text
best.pt
```

and continues training from it.

---

# Predict on Test Images

Run:

```bash
python3 scripts/04_predict_on_test_images.py
```

The script automatically:

- finds latest `best.pt`
- predicts on test images
- saves annotated predictions
- saves prediction `.txt` files

---

# Prediction Logic

```text
Image
 ↓
YOLO prediction
 ↓
Find reading window
 ↓
Ignore digits outside the reading window
 ↓
Keep digits inside the reading window only
 ↓
Sort digits left to right
 ↓
Create final meter reading
 ↓
If unknown digit exists:
Ask user to retake image
```

---

# Labeling Guidelines

- Tight bounding boxes improve accuracy
- Poor labels reduce model quality
- Window labels must cover only the reading window
- More diverse images improve generalization
- Strong rotation may reduce accuracy
- Images close to horizontal usually perform better
- Remove blurry or unusable images
- If labeling a digit may confuse the model, it is better not to label it than to force an incorrect label
- If training metrics are poor, optimize the labeling before retraining

---

# Future Work

Planned improvements:

- segmentation-based reading
- webcam live inference
- mobile deployment
- MQTT/cloud integration
- real-time validation
- automatic reading reconstruction

---

# Author

Gabriel BAZIRAMWABO  
Rwanda Coding Academy | Benax Technologies  
May 2026
