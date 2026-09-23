import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\tkg91\PycharmProjects\Apple_Detection_Capstone"
)

DATASET = PROJECT / "orchard_maturity"

MODEL_PATH = (
    PROJECT
    / "models"
    / "orchard_maturity_resnet50_best.pth"
)

RESULTS = (
    PROJECT
    / "results"
    / "classification"
)

RESULTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    DATASET / "test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

class_names = test_dataset.classes

print("\nClasses:", class_names)
print("Test images:", len(test_dataset))


# ============================================================
# LOAD RESNET50
# ============================================================

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded:")
print(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
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


print("\n========================================")
print("ORCHARD MATURITY TEST RESULTS")
print("========================================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {precision:.4f}")
print(f"Macro Recall:      {recall:.4f}")
print(f"Macro F1:          {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    RESULTS
    / "orchard_maturity_classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "ORCHARD MATURITY CLASSIFICATION RESULTS\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        f"Test Images: {len(test_dataset)}\n"
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
        "Classification Report:\n"
    )

    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Orchard Apple Maturity Confusion Matrix")

plt.tight_layout()

cm_path = (
    RESULTS
    / "orchard_maturity_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print("Report:")
print(report_path)

print("\nConfusion matrix:")
print(cm_path)