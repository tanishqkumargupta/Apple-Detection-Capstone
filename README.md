# 🍎 Apple Detection, Counting and Classification Using Deep Learning

A deep-learning-based computer vision framework for automated **apple detection, counting, ripeness classification, and orchard maturity analysis**.

The project combines YOLO11n for apple detection and counting with ResNet50-based classification models for fruit condition and orchard maturity analysis. EfficientNet-B0 is additionally evaluated as a comparison model.

---

## 📌 Project Overview

Accurate identification and monitoring of apples in orchard environments can support agricultural yield estimation and fruit-quality assessment.

This project investigates a modular computer vision pipeline consisting of:

- Apple detection using YOLO11n
- Apple counting from detected instances
- Raw/Ripe/Rotten classification using ResNet50
- ResNet50 vs EfficientNet-B0 comparison
- Orchard apple maturity classification
- Cross-domain detection evaluation
- End-to-end inference demonstration

The experiments use separate datasets for detection, ripeness classification, and orchard maturity classification.

---

## 🏗️ System Architecture

The primary inference pipeline is:

```text
Input Orchard Image
        │
        ▼
   YOLO11n Detector
        │
        ▼
 Apple Bounding Boxes
        │
        ├──────────────► Apple Count
        │
        ▼
  Apple Crop Extraction
        │
        ▼
     ResNet50
        │
        ▼
 ┌──────┼─────────┐
 │      │         │
Raw    Ripe     Rotten
A separate experiment investigates orchard maturity:

Orchard Apple Image
        │
        ▼
Apple Region Extraction
        │
        ▼
     ResNet50
        │
        ▼
Immature / Semi-mature / Mature

The complete architecture is available at:

results/architecture/system_architecture.png
📂 Project Structure
Apple_Detection_Capstone/
│
├── apple_ripeness/
│   ├── train/
│   ├── val/
│   └── test/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── orchard_maturity/
│   ├── train/
│   ├── val/
│   └── test/
│
├── minneapple/
│   └── detection/
│
├── models/
│   ├── apple_detection_best.pt
│   ├── apple_detection_last.pt
│   ├── resnet50_best.pth
│   ├── efficientnet_b0_best.pth
│   ├── orchard_maturity_resnet50_best.pth
│   └── orchard_maturity_resnet50_last.pth
│
├── src/
│   ├── detection.py
│   ├── classification.py
│   ├── pipeline.py
│   ├── evaluation.py
│   ├── classification_evaluation.py
│   ├── orchard_maturity_pipeline.py
│   ├── evaluate_orchard_maturity.py
│   ├── debug_orchard_detection.py
│   ├── crop_diagnostic.py
│   └── padded_diagnostic.py
│
├── scripts/
│   ├── convert_to_yolo.py
│   ├── create_ripeness_dataset.py
│   ├── create_orchard_maturity_dataset.py
│   ├── inspect_masks.py
│   ├── validate_masks.py
│   └── visualize_yolo.py
│
├── results/
│   ├── architecture/
│   │   └── system_architecture.png
│   ├── detection/
│   ├── classification/
│   ├── combined/
│   ├── FINAL_RESULTS.md
│   ├── RESULTS_TABLES.md
│   ├── RESULTS_AND_DISCUSSION.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENTAL_SETUP.md
│   ├── REFERENCES_AND_RESEARCH_BASIS.md
│   └── LIMITATIONS_AND_FUTURE_WORK.md
│
├── test_images/
├── data.yaml
├── requirements.txt
└── README.md
📊 Datasets
1. MinneApple

Used for:

Apple detection
Apple counting

The original instance masks were converted into YOLO-compatible bounding-box annotations.

Dataset split
Split	Images
Training	536
Validation	134
Total	670

The detection task contains one class:

apple

The validation set contains 5,660 annotated apple instances.

2. Fruit Ripeness Dataset

Used for:

Raw/Ripe/Rotten apple classification
ResNet50 vs EfficientNet-B0 comparison

Only apple images were selected from the original fruit dataset.

The class mapping used in this project is:

Original Category	Project Class
Unripe apple	Raw
Fresh apples	Ripe
Rotten apples	Rotten
Dataset distribution
Class	Total
Raw	2,305
Ripe	2,088
Rotten	2,943
Total	7,336
Split	Images
Training	4,777
Validation	1,192
Test	1,367
3. Multi-Stage Apple Dataset

Used for:

Orchard maturity classification
Cross-domain detection evaluation

The dataset contains:

Immature
Semi-mature
Mature

After excluding one unknown annotation:

Split	Images
Training	1,981
Validation	185
Test	408
Total	2,574
🤖 Models
YOLO11n

YOLO11n is used for single-class apple detection.

Input: 640 × 640
Classes: 1
Epochs: 50
Batch Size: 16

Model:

models/apple_detection_best.pt
ResNet50

ImageNet-pretrained ResNet50 is used for:

Raw/Ripe/Rotten classification
Orchard maturity classification

For ripeness classification:

Input: 224 × 224
Classes: 3
Epochs: 20
Batch Size: 32
Learning Rate: 0.0001
Optimizer: AdamW
Loss: Cross-Entropy

Model:

models/resnet50_best.pth
EfficientNet-B0

EfficientNet-B0 is used as a comparison architecture for the Raw/Ripe/Rotten classification task.

Input: 224 × 224
Classes: 3
Epochs: 20
Batch Size: 32
Learning Rate: 0.0001
Optimizer: AdamW
Loss: Cross-Entropy

Model:

models/efficientnet_b0_best.pth
📈 Experimental Results
Apple Detection
Metric	Result
Precision	84.8%
Recall	75.7%
mAP@50	84.7%
mAP@50–95	42.2%
Apple Counting
Metric	Result
Ground Truth	5,660
Predictions	5,901
MAE	4.5448
RMSE	6.4779
Counting Accuracy	89.24%
Raw/Ripe/Rotten Classification
ResNet50
Metric	Result
Accuracy	99.93%
Macro Precision	99.94%
Macro Recall	99.92%
Macro F1	99.93%
EfficientNet-B0
Metric	Result
Accuracy	99.93%
Macro Precision	99.91%
Macro Recall	99.92%
Macro F1	99.91%
Model Comparison
Model	Accuracy	Macro Precision	Macro Recall	Macro F1
ResNet50	99.93%	99.94%	99.92%	99.93%
EfficientNet-B0	99.93%	99.91%	99.92%	99.91%

ResNet50 was selected as the primary ripeness classifier because of its marginally higher macro precision and macro F1-score.

🌳 Orchard Maturity Classification

The separate orchard maturity experiment uses:

Immature
Semi-mature
Mature
ResNet50 Results
Metric	Result
Accuracy	88.48%
Macro Precision	89.51%
Macro Recall	88.45%
Macro F1	88.18%

The semi-mature category was more challenging to classify, reflecting the visual similarity between intermediate maturity stages.

🔄 Cross-Domain Evaluation

The MinneApple-trained YOLO11n detector was directly evaluated on the separate orchard maturity dataset without additional detector training.

Metric	Result
Ground Truth Apples	408
Predicted Apples	240
True Positives	5
False Positives	235
False Negatives	403
Precision	2.08%
Recall	1.23%
F1-score	1.54%

This experiment demonstrates a substantial domain shift between the two orchard datasets.

This is treated as an exploratory transfer experiment and not as the primary YOLO11n detection benchmark.

🔬 End-to-End Demonstration

The trained YOLO11n and ResNet50 models were connected into an inference pipeline:

Image
 ↓
YOLO11n
 ↓
Apple Detection
 ↓
Counting
 ↓
Crop Extraction
 ↓
ResNet50
 ↓
Raw / Ripe / Rotten

On a representative MinneApple orchard image:

Total detected: 101
Raw: 101
Ripe: 0
Rotten: 0

This result demonstrates the operation of the complete pipeline.

It is not reported as an end-to-end accuracy measurement, because the detection and ripeness models were trained using different datasets and therefore have a domain gap.

🖼️ Results

Representative outputs are available in:

results/detection/
results/classification/
results/combined/

Important visualizations include:

results/detection/20150919_174151_image1_detected.jpg

results/combined/20150919_174151_image1_classified.jpg

results/combined/orchard_maturity_pipeline.jpg

results/classification/confusion_matrix(2).png

results/classification/orchard_maturity_confusion_matrix.png

The system architecture is available at:

results/architecture/system_architecture.png
🛠️ Technologies Used
Python
PyTorch
Ultralytics YOLO
YOLO11n
ResNet50
EfficientNet-B0
OpenCV
NumPy
scikit-learn
Matplotlib
Pillow
⚙️ Installation

Clone the repository:

git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Apple_Detection_Capstone

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
Apple Detection
python src/detection.py
Ripeness Classification
python src/classification.py
End-to-End Detection + Classification
python src/pipeline.py
Ripeness Classification Evaluation
python src/classification_evaluation.py
Orchard Maturity Pipeline
python src/orchard_maturity_pipeline.py
Orchard Maturity Evaluation
python src/evaluate_orchard_maturity.py
📁 Trained Models

The trained model checkpoints are stored in:

models/
Task	Model
Apple Detection	apple_detection_best.pt
Apple Detection Last Checkpoint	apple_detection_last.pt
Raw/Ripe/Rotten	resnet50_best.pth
EfficientNet Comparison	efficientnet_b0_best.pth
Orchard Maturity	orchard_maturity_resnet50_best.pth
Orchard Maturity Last Checkpoint	orchard_maturity_resnet50_last.pth
⚠️ Limitations

The primary limitation identified during experimentation is domain shift.

The detection dataset contains orchard imagery, while the ripeness classification dataset primarily contains fruit-level images. Consequently, the high ripeness classification performance should not be interpreted as equivalent real-world orchard ripeness accuracy.

Similarly, the MinneApple-trained detector showed substantially reduced performance when transferred directly to the separate orchard maturity dataset.

Other limitations include:

Small and partially occluded apples
Variable illumination
Dense foliage and background clutter
Differences between image acquisition environments
Separate label spaces for ripeness and orchard maturity
🚀 Future Work

Potential future improvements include:

Training an orchard-specific detector
Domain adaptation between orchard datasets
Development of a unified orchard Raw/Ripe/Rotten dataset
Improved small-object detection
Lightweight model optimization
Larger multi-orchard evaluation
Real-time edge deployment
📚 Research Basis

This project was developed with reference to research in agricultural computer vision, particularly work involving deep-learning-based agricultural image classification and YOLO-based fruit detection.

The two primary reference studies are:

Attri, I., Awasthi, L. K., & Sharma, T. P. (2026).
DeepWave Feature Extractor (DW-FE): An Approach to Image Feature Extraction Combining Wavelet Transformation and Deep Learning for Mango and Apple Leaf Disease Classification.
Applied Fruit Science, 68, 298.
DOI: 10.1007/s10341-026-02009-6
Chouhan, S. S., Saxena, E., Shukla, A., Patel, R. K., & Singh, U. P. (2025).
Optimizing YOLO-Based Models for Real-Time Guava Detection with Probabilistic Fused Wiener Filter-Enhanced Feature Fusion.
Applied Fruit Science, 67, 383.
DOI: 10.1007/s10341-025-01613-2

The specific DW-FE and PFWF architectures from these studies were not implemented in this project. They were used as research and methodological references.

👨‍💻 Project Status

Status: Completed Experimental Prototype

The project currently includes:

✅ Apple detection
✅ Apple counting
✅ Raw/Ripe/Rotten classification
✅ ResNet50 vs EfficientNet-B0 comparison
✅ Orchard maturity classification
✅ Cross-domain evaluation
✅ End-to-end inference demonstration
✅ Trained model checkpoints
✅ Evaluation metrics
✅ Confusion matrices
✅ System architecture
✅ Research documentationcla