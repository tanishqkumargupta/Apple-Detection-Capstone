# Apple Detection, Counting, and Ripeness Classification

A modular deep-learning framework that detects individual apples in orchard imagery, classifies each detected apple as **raw / ripe / rotten**, and includes supporting analyses for apple counting, orchard maturity classification, and cross-domain generalization.

This repository accompanies the paper *"A Modular Deep Learning Framework for Apple Detection and Ripeness Classification, with Supporting Counting and Cross-Domain Generalization Evaluations"* (Patel, Gupta, Patil, Parmar, Agrawal, Pathak).

## Overview

Most published systems evaluate fruit detection, counting, and ripeness/maturity classification separately, each on a dataset built for that one task — leaving cross-dataset transferability largely unexamined. This project builds a **core pipeline** (detection → classification) and then probes its reach with three supporting experiments:

1. **Apple detection** — YOLO11n trained on MinneApple
2. **Ripeness classification** — ResNet50 (and a comparative EfficientNet-B0) trained on a fruit-level raw/ripe/rotten dataset
3. **Apple counting** — derived directly from detector output, no extra model
4. **Orchard maturity classification** — an independently trained ResNet50 on a separate, pixel-level annotated orchard dataset (immature / semi-mature / mature)
5. **Cross-domain detector transfer** — the MinneApple-trained detector applied, without retraining, to the orchard maturity dataset, to test generalization across orchard-imagery sources

```
Apple Image → YOLO11n Detection → Crop Detected Apples → ResNet50 Classification → Raw / Ripe / Rotten
                     │
                     └──► Apple Counting (from detections)

Orchard Images → ResNet50 (independently trained) → Immature / Semi-mature / Mature

MinneApple-trained YOLO11n (no retraining) → applied to Orchard Images → Cross-Domain Transfer Evaluation
```

## Key Results

| Component | Dataset | Model | Metric | Result |
|---|---|---|---|---|
| Detection (core) | MinneApple | YOLO11n | mAP@50 | 84.7% |
| Detection (core) | MinneApple | YOLO11n | Precision / Recall | 84.8% / 75.7% |
| Ripeness classification (core) | Fruit Ripeness | ResNet50 | Accuracy | 99.93% |
| Ripeness classification (core) | Fruit Ripeness | EfficientNet-B0 | Accuracy | 99.93% |
| Counting (supporting) | MinneApple | YOLO11n-derived | Counting accuracy | 89.24% (MAE 4.54, RMSE 6.48) |
| Maturity classification (supporting) | Multi-Stage orchard dataset | ResNet50 | Accuracy | 88.48% |
| Cross-domain detection (supporting) | Multi-Stage orchard dataset (transfer) | YOLO11n (MinneApple weights) | Precision / Recall / F1 | 2.08% / 1.23% / 1.54% |

The datasets are non-overlapping and evaluated separately — this table summarizes independently obtained results, not a single unified benchmark.

**Headline finding:** strong in-domain detection performance (84.7% mAP@50 on MinneApple) does not transfer across orchard-imagery sources — applying the same detector, unmodified, to a second orchard dataset collapses performance to 1.54% F1-score. Domain compatibility needs explicit evaluation when composing agricultural computer-vision pipelines from independently trained models.

## Datasets

| Dataset | Role | Size | Classes | Annotation | Source |
|---|---|---|---|---|---|
| **MinneApple** | Core detection + counting | 670 images / 5,660 val. instances | 1 (apple) | Instance masks → bounding boxes | [University of Minnesota Data Repository](https://doi.org/10.13020/8ecp-3r13) |
| **Fruit Ripeness (apple subset)** | Core ripeness classification | 7,336 images | 3 (raw / ripe / rotten) | Folder-based labels | [Kaggle](https://www.kaggle.com/) — "Fruit Ripeness: Unripe, Ripe, and Rotten" |
| **Multi-Stage Apple Dataset** | Supporting: maturity classification + cross-domain detection | 2,574 instances | 3 (immature / semi-mature / mature) | Pixel-level polygon (VIA format) | [Mendeley Data](https://doi.org/10.17632/gfcmdbvw65.4) |

Datasets are not redistributed in this repository — download from the sources above and place them under `data/` (see `data/README.md` for expected structure).

## Pipeline Details

### 1. Detection — YOLO11n
- Input size: 640×640, batch size 16, 50 epochs, single class (`apple`)
- Trained on MinneApple (536 train / 134 val images) on an NVIDIA Tesla T4
- Confidence threshold: 0.25
- Best checkpoint by validation mAP@50: `apple_detection_best.pt`

### 2. Ripeness Classification — ResNet50 / EfficientNet-B0
- ImageNet-pretrained backbones, final layer replaced with a 3-way head (raw / ripe / rotten)
- Input 224×224, batch size 32, 20 epochs, AdamW (lr 1e-4, weight decay 1e-4)
- Augmentation (train only): horizontal flip, ±10° rotation, brightness/contrast/saturation jitter
- Best checkpoint by validation accuracy: `resnet50_best.pth`

### 3. Apple Counting
- Count = number of retained detections per image (confidence ≥ 0.25)
- No additional regression model — evaluated with MAE, RMSE, and a normalized counting-accuracy metric

### 4. Orchard Maturity Classification
- Independently trained ResNet50 on the Multi-Stage dataset's polygon-cropped apple regions
- Separate label space (immature / semi-mature / mature); not part of the core pipeline
- Best checkpoint: `orchard_maturity_resnet50_best.pth`

### 5. Cross-Domain Detection Evaluation
- The MinneApple-trained detector (unmodified) is run on the Multi-Stage dataset's 239 test images
- Matching by IoU ≥ 0.50 against ground-truth polygon boxes
- Exploratory only — quantifies detector transfer, not a general detector benchmark

### 6. End-to-End Demonstration
- YOLO11n detection → crop extraction → ResNet50 ripeness classification, run sequentially on a representative MinneApple image
- Qualitative only: no dataset in this study provides ground-truth ripeness labels for individually detected MinneApple apples, so no end-to-end accuracy is reported

## Repository Structure

```
.
├── Capstone.ipynb          # End-to-end notebook: setup, detection, classification, counting,
│                           # maturity classification, cross-domain evaluation, demo pipeline
├── data/                   # (not included) place downloaded datasets here
├── models/                 # (not included) trained checkpoints
│   ├── apple_detection_best.pt
│   ├── resnet50_best.pth
│   ├── efficientnet_b0_best.pth
│   └── orchard_maturity_resnet50_best.pth
└── README.md
```

## Getting Started

```bash
git clone https://github.com/tanishqkumargupta/Apple-Detection-Capstone.git
cd Apple-Detection-Capstone
pip install ultralytics torch torchvision scikit-learn matplotlib seaborn
```

The notebook (`Capstone.ipynb`) was developed on Google Colab with a Tesla T4 GPU and walks through:
1. Environment setup
2. Dataset preparation (MinneApple → YOLO format; ripeness dataset; Multi-Stage polygon → bbox conversion)
3. YOLO11n training and evaluation
4. Apple counting from detections
5. ResNet50 / EfficientNet-B0 ripeness classifier training and comparison
6. Orchard maturity classifier training
7. Cross-domain detection evaluation
8. Crop extraction and qualitative end-to-end demo

## Limitations

- **Dataset-domain mismatch**: the ripeness dataset is fruit-level (close-up) imagery, while MinneApple is in-orchard imagery — the 99.93% ripeness figure characterizes fruit-level classification, not in-orchard ripeness assessment.
- **No quantitative end-to-end benchmark**: no dataset provides both apple-detection annotations and ripeness labels for the same images.
- **Severe cross-domain degradation**: the detector does not transfer between the two orchard-imagery sources without retraining or domain adaptation.
- **No deployment/efficiency measurements**: inference speed, model size trade-offs, and field/real-time performance were not measured.

See the paper's Section 7 for the full list.

## Future Work

- Fine-tune/retrain the detector on target-orchard imagery
- Explore domain-adaptation techniques (e.g., CycleGAN-based approaches) to reduce the cross-domain gap
- Build a unified dataset with both orchard-scene detection and per-apple ripeness labels for genuine end-to-end evaluation
- Higher-resolution / multi-scale detection for small and occluded apples
- Benchmark additional lightweight architectures for edge deployment
- Evaluate across more orchards, cultivars, viewpoints, and lighting conditions
- Joint/multi-task training in place of the sequential detect-then-classify pipeline

## Citation

If you use this work, please cite:

```
Patel, R.K., Gupta, T.K., Patil, S.U., Parmar, S., Agrawal, R., Pathak, V.S.
A Modular Deep Learning Framework for Apple Detection and Ripeness Classification,
with Supporting Counting and Cross-Domain Generalization Evaluations.
```

## Data Availability

- MinneApple: University of Minnesota Data Repository, DOI [10.13020/8ecp-3r13](https://doi.org/10.13020/8ecp-3r13)
- Fruit Ripeness (Unripe, Ripe, Rotten): Kaggle
- Multi-Stage Pixel-Level Annotated Apple Dataset: Mendeley Data, DOI [10.17632/gfcmdbvw65.4](https://doi.org/10.17632/gfcmdbvw65.4)

## License / Funding / Conflicts

The authors received no financial support for this research. No competing interests are declared. This study used only publicly available image datasets and did not involve human participants, animals, or identifiable personal data.
