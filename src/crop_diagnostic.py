from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "apple_detection_best.pt"
RESNET_MODEL_PATH = BASE_DIR / "models" / "resnet50_best.pth"

INPUT_IMAGE = BASE_DIR / "test_images" / "20150919_174151_image1.png"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "raw",
    "ripe",
    "rotten"
]


# ============================================================
# TRANSFORM
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
# LOAD YOLO
# ============================================================

from ultralytics import YOLO

print("Loading YOLO...")
yolo_model = YOLO(str(YOLO_MODEL_PATH))


# ============================================================
# LOAD RESNET50
# ============================================================

print("Loading ResNet50...")

resnet_model = models.resnet50(weights=None)

resnet_model.fc = nn.Linear(
    2048,
    3
)

checkpoint = torch.load(
    RESNET_MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    resnet_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    resnet_model.load_state_dict(checkpoint)

resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

print("Both models loaded.")


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(str(INPUT_IMAGE))

if image is None:
    raise FileNotFoundError(INPUT_IMAGE)


# ============================================================
# YOLO
# ============================================================

results = yolo_model(
    image,
    imgsz=640,
    conf=0.25,
    verbose=False
)

boxes = results[0].boxes.xyxy.cpu().numpy()

print(f"\nDetected apples: {len(boxes)}")


# ============================================================
# DIAGNOSTIC VARIABLES
# ============================================================

crop_widths = []
crop_heights = []

prediction_counts = {
    "raw": 0,
    "ripe": 0,
    "rotten": 0
}

raw_confidences = []
ripe_confidences = []
rotten_confidences = []


# ============================================================
# CLASSIFY EACH CROP
# ============================================================

for i, box in enumerate(boxes):

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    width = x2 - x1
    height = y2 - y1

    crop_widths.append(width)
    crop_heights.append(height)

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(crop_rgb)

    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = resnet_model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    label = CLASS_NAMES[predicted_class]

    confidence = probabilities[predicted_class].item()

    prediction_counts[label] += 1

    if label == "raw":
        raw_confidences.append(confidence)

    elif label == "ripe":
        ripe_confidences.append(confidence)

    else:
        rotten_confidences.append(confidence)

    # Print first 20
    if i < 20:

        print(
            f"\nApple {i + 1}"
        )

        print(
            f"Crop size: {width} x {height} px"
        )

        print(
            f"Raw:    {probabilities[0].item():.4f}"
        )

        print(
            f"Ripe:   {probabilities[1].item():.4f}"
        )

        print(
            f"Rotten: {probabilities[2].item():.4f}"
        )

        print(
            f"Prediction: {label} "
            f"({confidence:.4f})"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CROP DIAGNOSTIC SUMMARY")
print("=" * 60)

print(f"Total crops: {len(crop_widths)}")

if crop_widths:

    avg_width = sum(crop_widths) / len(crop_widths)
    avg_height = sum(crop_heights) / len(crop_heights)

    print(
        f"Average crop size: "
        f"{avg_width:.2f} x {avg_height:.2f} px"
    )

    print(
        f"Smallest crop: "
        f"{min(crop_widths)} x {min(crop_heights)} px"
    )

    print(
        f"Largest crop: "
        f"{max(crop_widths)} x {max(crop_heights)} px"
    )


print("\nPredictions:")

print(
    f"Raw:    {prediction_counts['raw']}"
)

print(
    f"Ripe:   {prediction_counts['ripe']}"
)

print(
    f"Rotten: {prediction_counts['rotten']}"
)


if raw_confidences:

    avg_raw = (
        sum(raw_confidences)
        / len(raw_confidences)
    )

    print(
        f"\nAverage RAW confidence: "
        f"{avg_raw:.4f}"
    )


print("\nDiagnostic complete.")