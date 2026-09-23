from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "resnet50_best.pth"
DATASET_DIR = BASE_DIR / "apple_ripeness" / "test"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# IMAGE TRANSFORMATION
# Same preprocessing used during evaluation
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
# LOAD RESNET50
# ============================================================

print("Loading ResNet50...")

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

# Handle both possible checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)


model = model.to(DEVICE)
model.eval()

print("ResNet50 loaded successfully.")
print(f"Model: {MODEL_PATH}")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_label = CLASS_NAMES[predicted_class]

    confidence = probabilities[0][predicted_class].item()

    return predicted_label, confidence


# ============================================================
# TEST ONE IMAGE FROM EACH CLASS
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 CLASSIFICATION TEST")
print("=" * 60)


for class_name in CLASS_NAMES:

    class_dir = DATASET_DIR / class_name

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }
    ]

    if not images:
        print(f"No images found for class: {class_name}")
        continue

    image_path = images[0]

    prediction, confidence = predict_image(
        image_path
    )

    print(f"\nActual      : {class_name}")
    print(f"Image       : {image_path.name}")
    print(f"Prediction  : {prediction}")
    print(f"Confidence  : {confidence:.4f}")


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)