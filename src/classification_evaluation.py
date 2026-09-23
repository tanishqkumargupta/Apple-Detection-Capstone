from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "resnet50_best.pth"
TEST_DIR = BASE_DIR / "apple_ripeness" / "test"

RESULTS_DIR = BASE_DIR / "results" / "classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "raw",
    "ripe",
    "rotten"
]


# ============================================================
# TRANSFORMATION
# SAME AS ORIGINAL MODEL EVALUATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Classes: {test_dataset.classes}")
print(f"Test images: {len(test_dataset)}")


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

print(
    f"Class mapping: {test_dataset.class_to_idx}"
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("\nLoading ResNet50...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    2048,
    3
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)
model.eval()

print("ResNet50 loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning evaluation...")

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 TEST RESULTS")
print("=" * 60)

print(
    f"Test images       : {len(all_labels)}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Macro Precision   : {precision:.4f}"
)

print(
    f"Macro Recall      : {recall:.4f}"
)

print(
    f"Macro F1          : {f1:.4f}"
)


print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)

print(report)


print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

print(cm)


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

report_path = (
    RESULTS_DIR /
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("RESNET50 TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test images: {len(all_labels)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision: {precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {f1:.4f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write("=" * 60 + "\n")

    f.write(report)

    f.write("\nCONFUSION MATRIX\n")
    f.write("=" * 60 + "\n")

    f.write(
        np.array2string(cm)
    )


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ResNet50 - Apple Ripeness Classification"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)


# Add numbers inside cells
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()


cm_path = (
    RESULTS_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    f"Report: {report_path}"
)

print(
    f"Confusion matrix: {cm_path}"
)

print("\nEvaluation complete.")