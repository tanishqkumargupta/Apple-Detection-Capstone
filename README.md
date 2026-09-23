# Apple Detection, Counting and Ripeness Classification Using Deep Learning

A modular deep-learning framework for apple detection, counting, ripeness classification, orchard maturity analysis, and cross-domain evaluation in agricultural computer vision.

The project combines **YOLO11n** for apple detection with **ResNet50** for raw/ripe/rotten classification. An **EfficientNet-B0** model is additionally evaluated under matched training conditions. Separate experiments investigate orchard maturity classification and detector transfer across orchard datasets.

This repository accompanies the paper *"A Modular Deep Learning Framework for Apple Detection and Ripeness Classification, with Supporting Counting and Cross-Domain Generalization Evaluations"* (Patel, Gupta, Patil, Parmar, Agrawal, Pathak).

> **Note:** This repository contains the implementation code, trained model checkpoints, configuration files, and representative test images. The original datasets and generated experimental result files are **not** included.

## Overview

Automated apple analysis in orchard environments is challenging because of:

- Dense fruit clusters
- Partial occlusion by leaves and branches
- Variable illumination
- Small or distant apples
- Background clutter
- Differences between orchard datasets and acquisition conditions

This project investigates a modular framework consisting of:

1. Apple detection using YOLO11n
2. Apple counting derived directly from detector outputs
3. Raw/Ripe/Rotten classification using ResNet50
4. ResNet50 vs. EfficientNet-B0 comparison
5. Orchard apple maturity classification
6. Cross-domain detector-transfer evaluation
7. Qualitative end-to-end detection-to-ripeness inference

The experiments use separate datasets for detection, fruit-level ripeness classification, and orchard maturity classification — they are reported independently rather than as a single unified benchmark.

## System Architecture

### Core Detection and Ripeness Pipeline

```
                 Input Orchard Image
                         │
                         ▼
                   YOLO11n Detector
                         │
                         ▼
                Apple Bounding Boxes
                    │           │
                    │           └──────────► Apple Count
                    │
                    ▼
              Apple Crop Extraction
                         │
                         ▼
                    ResNet50
                         │
                ┌────────┼────────┐
                ▼        ▼        ▼
              Raw      Ripe     Rotten
```

### Orchard Maturity Experiment

```
          Orchard Apple Image
                  │
                  ▼
         Apple Region Extraction
                  │
                  ▼
               ResNet50
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    Immature  Semi-mature   Mature
```

### Cross-Domain Detector Evaluation

```
MinneApple-trained YOLO11n
             │
             │ No retraining
             ▼
   Multi-Stage Orchard Dataset
             │
             ▼
    Detection Transfer Analysis
```

The detection and ripeness models were trained on different datasets. Therefore, the combined detection-to-ripeness pipeline is demonstrated **qualitatively** rather than evaluated with a quantitative end-to-end accuracy metric.

## Repository Structure

```
Apple_Detection_Capstone/
│
├── models/
│   ├── apple_detection_best.pt
│   ├── apple_detection_last.pt
│   ├── efficientnet_b0_best.pth
│   ├── orchard_maturity_resnet50_best.pth
│   ├── orchard_maturity_resnet50_last.pth
│   ├── resnet50_best.pth
│   └── resnet50_last.pth
│
├── scripts/
│   ├── convert_to_yolo.py
│   ├── create_orchard_maturity_dataset.py
│   ├── create_ripeness_dataset.py
│   ├── evaluate_orchard_maturity.py
│   ├── inspect_masks.py
│   ├── train_orchard_maturity.py
│   ├── validate_masks.py
│   ├── visualize_orchard_maturity.py
│   └── visualize_yolo.py
│
├── src/
│   ├── classification.py
│   ├── classification_evaluation.py
│   ├── crop_diagnostic.py
│   ├── debug_orchard_detection.py
│   ├── detection.py
│   ├── evaluate_orchard_detection.py
│   ├── orchard_maturity_pipeline.py
│   ├── padded_diagnostic.py
│   └── pipeline.py
│
├── test_images/
│
├── data.yaml
├── .gitignore
├── .gitattributes
└── README.md
```

The datasets, virtual environment, generated result files, and intermediate dataset directories are intentionally excluded from the repository.

## Datasets

### 1. MinneApple

**Role:** Apple detection · Apple counting

The original MinneApple instance masks were converted into single-class YOLO-compatible bounding-box annotations.

| Split | Images |
|---|---|
| Training | 536 |
| Validation | 134 |
| **Total** | **670** |

The detection task contains one class: `apple`. The validation split contains 5,660 annotated apple instances.

### 2. Fruit Ripeness Dataset

**Role:** Raw/Ripe/Rotten apple classification · ResNet50 vs. EfficientNet-B0 comparison

Only apple images from the public fruit-ripeness dataset were used. The original categories were mapped as follows:

| Original Category | Project Class |
|---|---|
| Unripe apple | Raw |
| Fresh apples | Ripe |
| Rotten apples | Rotten |

Dataset distribution:

| Class | Total |
|---|---|
| Raw | 2,305 |
| Ripe | 2,088 |
| Rotten | 2,943 |
| **Total** | **7,336** |

| Split | Images |
|---|---|
| Training | 4,777 |
| Validation | 1,192 |
| Test | 1,367 |

The ripeness dataset primarily contains fruit-level images rather than apples embedded in orchard scenes.

### 3. Multi-Stage Apple Dataset

**Role:** Orchard maturity classification · Cross-domain detection evaluation

The dataset provides pixel-level annotations for three orchard maturity categories: Immature, Semi-mature, Mature. After excluding one annotation with an unresolved class label, 2,574 valid apple instances were used.

| Split | Valid Apple Instances |
|---|---|
| Training | 1,981 |
| Validation | 185 |
| Test | 408 |
| **Total** | **2,574** |

This dataset is independently collected from MinneApple and differs in orchard layout and image acquisition conditions, making it suitable for the supporting cross-domain evaluation.

## Models

### YOLO11n — Apple Detection

| Configuration | Value |
|---|---|
| Model | YOLO11n |
| Input size | 640 × 640 |
| Classes | 1 |
| Epochs | 50 |
| Batch size | 16 |
| Hardware | NVIDIA Tesla T4 |
| Detection class | Apple |

Checkpoint: `models/apple_detection_best.pt`

The detector uses a confidence threshold of 0.25 for retained detections.

### ResNet50 — Raw/Ripe/Rotten Classification

| Configuration | Value |
|---|---|
| Model | ResNet50 |
| Input size | 224 × 224 |
| Classes | 3 |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss | Cross-Entropy |

Classes: Raw · Ripe · Rotten

Checkpoint: `models/resnet50_best.pth`

### EfficientNet-B0 — Comparison Model

Trained under matched conditions (same input size, epochs, batch size, optimizer, learning rate, weight decay, loss) on the same ripeness dataset.

Checkpoint: `models/efficientnet_b0_best.pth`

### ResNet50 — Orchard Maturity Classification

A separate ImageNet-pretrained ResNet50 was trained **independently** for: Immature · Semi-mature · Mature.

Checkpoint: `models/orchard_maturity_resnet50_best.pth`

This classifier is not the same model as the raw/ripe/rotten classifier.

## Experimental Results

### Apple Detection

YOLO11n evaluation on the 134-image MinneApple validation split:

| Metric | Result |
|---|---|
| Precision | 84.8% |
| Recall | 75.7% |
| mAP@50 | 84.7% |
| mAP@50–95 | 42.2% |

### Apple Counting

Counting is derived directly from the detections produced by YOLO11n.

| Metric | Result |
|---|---|
| Evaluation images | 134 |
| Ground-truth apples | 5,660 |
| Predicted apples | 5,901 |
| MAE | 4.5448 |
| RMSE | 6.4779 |
| Counting accuracy | 89.24% |

The reported counting accuracy uses the normalized error formulation:

```
CountAcc = (1 - MAE / mean ground-truth count) × 100
```

### Raw/Ripe/Rotten Classification

**ResNet50**

| Metric | Result |
|---|---|
| Accuracy | 99.93% |
| Macro Precision | 99.94% |
| Macro Recall | 99.92% |
| Macro F1 | 99.93% |

**EfficientNet-B0**

| Metric | Result |
|---|---|
| Accuracy | 99.93% |
| Macro Precision | 99.91% |
| Macro Recall | 99.92% |
| Macro F1 | 99.91% |

The two models produced near-identical test accuracy on this dataset. ResNet50 was retained for the downstream qualitative detection-to-ripeness pipeline because it achieved marginally higher macro precision and macro F1. No statistical significance test was performed on this difference.

### Orchard Maturity Classification

| Metric | Result |
|---|---|
| Accuracy | 88.48% |
| Macro Precision | 89.51% |
| Macro Recall | 88.45% |
| Macro F1 | 88.18% |

The semi-mature category was the most difficult class, particularly because it represents an intermediate visual transition between immature and mature apples.

### Cross-Domain Detection

The YOLO11n detector trained on MinneApple was applied directly to the Multi-Stage dataset **without retraining or fine-tuning**.

| Metric | Result |
|---|---|
| Test images | 239 |
| Ground-truth apples | 408 |
| Predicted apples | 240 |
| True positives | 5 |
| False positives | 235 |
| False negatives | 403 |
| Precision | 2.08% |
| Recall | 1.23% |
| F1-score | 1.54% |

This experiment is treated as an exploratory cross-domain transfer evaluation rather than as the primary YOLO11n benchmark.

### End-to-End Demonstration

```
Input Image
    │
    ▼
YOLO11n Detection
    │
    ├──────────────► Apple Count
    │
    ▼
Apple Crop Extraction
    │
    ▼
224 × 224 Resize + Normalization
    │
    ▼
ResNet50 Ripeness Classification
    │
    ▼
Raw / Ripe / Rotten
```

On a representative MinneApple orchard image:

```
Detected apples: 101
Raw:             101
Ripe:              0
Rotten:            0
```

This demonstrates that the independently trained models can be connected into a working inference pipeline. It is a **qualitative** demonstration and is **not** an end-to-end accuracy measurement — the detection and ripeness models were trained on different datasets, and the study does not contain ground-truth raw/ripe/rotten labels for the individual apples detected in the MinneApple orchard images.

## Limitations

The primary limitation identified during experimentation is **domain shift**. The detection model is trained on orchard imagery, whereas the raw/ripe/rotten classifier is trained primarily on fruit-level images — the 99.93% fruit-level classification accuracy should not be interpreted as equivalent to orchard-wide ripeness classification accuracy. The cross-domain experiment further demonstrates that strong in-domain detection performance does not necessarily transfer to another orchard dataset.

Additional limitations include:

- Small and partially occluded apples
- Dense foliage and background clutter
- Variable illumination
- Differences in image acquisition conditions
- Differences between fruit-level and orchard-level imagery
- Separate label spaces for ripeness and orchard maturity
- Lack of a common dataset containing orchard detection annotations and reliable raw/ripe/rotten labels for the same apple instances
- No quantitative end-to-end ripeness accuracy
- No inference-speed or real-time deployment measurements

## Future Work

- Training an orchard-specific detector
- Domain adaptation between orchard datasets
- Development of a unified orchard raw/ripe/rotten dataset
- Improved small-object detection
- Lightweight model optimization
- Larger multi-orchard evaluation
- Quantitative end-to-end evaluation on a unified dataset
- Real-time edge deployment

## Technologies

- Python
- PyTorch
- Torchvision
- Ultralytics YOLO
- YOLO11n
- ResNet50
- EfficientNet-B0
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pillow

## Repository Usage

The repository contains trained checkpoints and implementation code. The scripts below depend on the corresponding datasets and local paths used during experimentation — the original datasets are not distributed with this repository.

**Detection**
```bash
python src/detection.py
```

**Raw/Ripe/Rotten Classification**
```bash
python src/classification.py
```

**End-to-End Detection + Classification**
```bash
python src/pipeline.py
```

**Classification Evaluation**
```bash
python src/classification_evaluation.py
```

**Orchard Maturity Pipeline**
```bash
python src/orchard_maturity_pipeline.py
```

**Orchard Maturity Evaluation**
```bash
python scripts/evaluate_orchard_maturity.py
```

**Cross-Domain Detection Evaluation**
```bash
python src/evaluate_orchard_detection.py
```

## Trained Checkpoints

| Task | Checkpoint |
|---|---|
| Apple detection | `models/apple_detection_best.pt` |
| Apple detection — last checkpoint | `models/apple_detection_last.pt` |
| Raw/Ripe/Rotten classification | `models/resnet50_best.pth` |
| Raw/Ripe/Rotten — last checkpoint | `models/resnet50_last.pth` |
| EfficientNet comparison | `models/efficientnet_b0_best.pth` |
| Orchard maturity classification | `models/orchard_maturity_resnet50_best.pth` |
| Orchard maturity — last checkpoint | `models/orchard_maturity_resnet50_last.pth` |

Model checkpoints are stored using **Git Large File Storage (Git LFS)** where required.

## Research Basis

The project was developed with reference to research in agricultural computer vision, particularly work involving deep-learning-based agricultural image classification and YOLO-based fruit detection.

**Primary Reference Studies**

Attri, I., Awasthi, L. K., & Sharma, T. P. (2026). *DeepWave Feature Extractor (DW-FE): An Approach to Image Feature Extraction Combining Wavelet Transformation and Deep Learning for Mango and Apple Leaf Disease Classification.* Applied Fruit Science, 68, 298. DOI: [10.1007/s10341-026-02009-6](https://doi.org/10.1007/s10341-026-02009-6)

Chouhan, S. S., Saxena, E., Shukla, A., Patel, R. K., & Singh, U. P. (2025). *Optimizing YOLO-Based Models for Real-Time Guava Detection with Probabilistic Fused Wiener Filter-Enhanced Feature Fusion.* Applied Fruit Science, 67, 383. DOI: [10.1007/s10341-025-01613-2](https://doi.org/10.1007/s10341-025-01613-2)

The DW-FE and PFWF architectures from these studies were **not** implemented in this project — they were used as research and methodological references.

## Data Availability

- MinneApple: University of Minnesota Data Repository, DOI [10.13020/8ecp-3r13](https://doi.org/10.13020/8ecp-3r13)
- Fruit Ripeness (Unripe, Ripe, Rotten): Kaggle
- Multi-Stage Pixel-Level Annotated Apple Dataset: Mendeley Data, DOI [10.17632/gfcmdbvw65.4](https://doi.org/10.17632/gfcmdbvw65.4)

## Project Status

**Status:** Completed Experimental Prototype

Implemented components:

- [x] Apple detection
- [x] Apple counting
- [x] Raw/Ripe/Rotten classification
- [x] ResNet50 vs. EfficientNet-B0 comparison
- [x] Orchard maturity classification
- [x] Cross-domain detection evaluation
- [x] End-to-end inference demonstration
- [x] Trained model checkpoints
- [x] Quantitative evaluation
- [x] Classification confusion matrices generated during experimentation
- [x] Research manuscript and experimental documentation maintained separately

## Funding / Conflicts of Interest

The authors received no financial support for this research. No competing interests are declared. This study used only publicly available image datasets and did not involve human participants, animals, or identifiable personal data.
