# Accident Detection and Intensity Classification

This project uses deep learning to **detect accidents in road surveillance images** and further **classify their intensity** as **strong** or **weak**.

## Features

1. **YOLOv8-based Accident Detection**
   - Trained on a custom dataset using Ultralytics YOLO.
   - Detects accident regions in images with bounding boxes.
   - Trained with heavy augmentations and multi-GPU support to prioritize **recall**.

2. **ResNet18-based Intensity Classifier**
   - Classifies cropped accident images (from YOLO detections) into:
     - **Strong Accident**
     - **Weak Accident**
   - Lightweight and fast, suitable for real-time applications.

## Model Architecture

- **YOLOv8 (custom-tuned)**: For object detection (accident localization).
- **ResNet18 (PyTorch)**: For image-level binary classification (accident intensity).

## Dataset Structure
```
Accident_Refined/
├── train/
│   ├── images/         # Training images
│   └── labels/         # Corresponding YOLO-format labels
├── valid/
│   ├── images/         # Validation images
│   └── labels/         # Validation labels
├── test/
│   ├── images/         # Test images
│   └── labels/         # Test labels
```

- Labels follow YOLO format.
- Intensity labels used for classification are separate and apply to detected regions.

## Inference Pipeline

1. **Input image** → YOLO model detects bounding boxes of accidents.
2. **Detected region** → Cropped and passed to ResNet18 classifier.
3. Output → Bounding box with **"Strong"** or **"Weak"** tag.

## Evaluation Metrics

| Metric         | Value   |
|----------------|---------|
| Precision      | ~0.94   |
| Recall         | TBD     |
| mAP@0.5        | TBD     |
| mAP@0.5:0.95   | TBD     |
| ResNet Accuracy| TBD     |

> Focused on **maximizing recall** to minimize missed detections.

## Requirements

- Python 3.11+
- PyTorch 2.6
- Ultralytics ≥ 8.3.0
- OpenCV, matplotlib, torchvision

Install dependencies:
```bash
pip install ultralytics torchvision opencv-python matplotlib
