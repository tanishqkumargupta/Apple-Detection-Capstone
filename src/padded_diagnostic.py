from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "apple_detection_best.pt"
RESNET_MODEL_PATH = BASE_DIR / "models" / "resnet50_best.pth"

INPUT_IMAGE = BASE_DIR / "test_images" / "20150919_174151_image1.png"

OUTPUT_DIR = BASE_DIR / "results" / "classification" / "padded_crops"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

print("Loading YOLO...")

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("Loading ResNet50...")

resnet_model = models.resnet50(
    weights=None
)

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

    resnet_model.load_state_dict(
        checkpoint
    )

resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

print("Both models loaded.")


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    str(INPUT_IMAGE)
)

if image is None:
    raise FileNotFoundError(
        INPUT_IMAGE
    )


height, width = image.shape[:2]


# ============================================================
# YOLO DETECTION
# ============================================================

results = yolo_model(
    image,
    imgsz=640,
    conf=0.25,
    verbose=False
)

boxes = results[0].boxes.xyxy.cpu().numpy()

print(
    f"\nDetected apples: {len(boxes)}"
)


# ============================================================
# PADDING
# ============================================================

# 2.0 means the crop becomes approximately
# twice as wide and twice as tall around the apple.

PADDING_FACTOR = 2.0


# ============================================================
# VARIABLES
# ============================================================

counts = {
    "raw": 0,
    "ripe": 0,
    "rotten": 0
}

crop_widths = []
crop_heights = []


# ============================================================
# PROCESS FIRST 20 APPLES
# ============================================================

for i, box in enumerate(boxes):

    if i >= 20:
        break

    x1, y1, x2, y2 = map(
        int,
        box
    )

    apple_width = x2 - x1
    apple_height = y2 - y1

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Padded dimensions
    new_width = int(
        apple_width * PADDING_FACTOR
    )

    new_height = int(
        apple_height * PADDING_FACTOR
    )

    new_x1 = center_x - new_width // 2
    new_y1 = center_y - new_height // 2

    new_x2 = center_x + new_width // 2
    new_y2 = center_y + new_height // 2

    # Keep inside image
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)

    new_x2 = min(width, new_x2)
    new_y2 = min(height, new_y2)

    crop = image[
        new_y1:new_y2,
        new_x1:new_x2
    ]

    if crop.size == 0:
        continue

    padded_width = new_x2 - new_x1
    padded_height = new_y2 - new_y1

    crop_widths.append(
        padded_width
    )

    crop_heights.append(
        padded_height
    )

    # BGR → RGB
    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        crop_rgb
    )

    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )

    # ========================================================
    # RESNET
    # ========================================================

    with torch.no_grad():

        output = resnet_model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    label = CLASS_NAMES[
        predicted_class
    ]

    confidence = probabilities[
        predicted_class
    ].item()

    counts[label] += 1

    # ========================================================
    # SAVE CROP
    # ========================================================

    crop_path = (
        OUTPUT_DIR /
        f"apple_{i + 1:02d}_{label}.jpg"
    )

    cv2.imwrite(
        str(crop_path),
        crop
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nApple {i + 1}"
    )

    print(
        f"Original box : "
        f"{apple_width} x {apple_height} px"
    )

    print(
        f"Padded crop  : "
        f"{padded_width} x {padded_height} px"
    )

    print(
        f"Raw          : "
        f"{probabilities[0].item():.4f}"
    )

    print(
        f"Ripe         : "
        f"{probabilities[1].item():.4f}"
    )

    print(
        f"Rotten       : "
        f"{probabilities[2].item():.4f}"
    )

    print(
        f"Prediction   : "
        f"{label} ({confidence:.4f})"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PADDED CROP DIAGNOSTIC")
print("=" * 60)

print(
    f"Images analyzed: "
    f"{len(crop_widths)}"
)

if crop_widths:

    average_width = (
        sum(crop_widths)
        / len(crop_widths)
    )

    average_height = (
        sum(crop_heights)
        / len(crop_heights)
    )

    print(
        f"Average padded crop: "
        f"{average_width:.2f} x "
        f"{average_height:.2f} px"
    )

print(
    f"\nRaw:    {counts['raw']}"
)

print(
    f"Ripe:   {counts['ripe']}"
)

print(
    f"Rotten: {counts['rotten']}"
)

print(
    f"\nSaved crops to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDiagnostic complete."
)