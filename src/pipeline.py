from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "apple_detection_best.pt"
RESNET_MODEL_PATH = BASE_DIR / "models" / "resnet50_best.pth"

INPUT_IMAGE = BASE_DIR / "test_images" / "20150919_174151_image1.png"

OUTPUT_DIR = BASE_DIR / "results" / "combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# RESNET TRANSFORM
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

print("Loading YOLO11n...")

yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)

print("YOLO11n loaded.")


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

print("ResNet50 loaded.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(INPUT_IMAGE)
)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {INPUT_IMAGE}"
    )

print(f"\nInput image: {INPUT_IMAGE.name}")


# ============================================================
# YOLO DETECTION
# ============================================================

results = yolo_model(
    image,
    imgsz=640,
    conf=0.25,
    verbose=False
)

result = results[0]

if result.boxes is None:
    print("No apples detected.")
    exit()

boxes = result.boxes.xyxy.cpu().numpy()

print(f"Detected apples: {len(boxes)}")


# ============================================================
# CLASSIFICATION COUNTS
# ============================================================

counts = {
    "raw": 0,
    "ripe": 0,
    "rotten": 0
}


# ============================================================
# PROCESS EACH APPLE
# ============================================================

for index, box in enumerate(boxes):

    x1, y1, x2, y2 = map(int, box)

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Crop apple
    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        continue

    # OpenCV BGR → RGB
    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        crop_rgb
    )

    # Transform
    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    # ResNet prediction
    with torch.no_grad():

        output = resnet_model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    label = CLASS_NAMES[
        predicted_class
    ]

    counts[label] += 1

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    text = f"{label} {confidence:.2f}"

    cv2.putText(
        image,
        text,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# SUMMARY
# ============================================================

total = sum(counts.values())

print("\n" + "=" * 60)
print("INTEGRATED PIPELINE RESULT")
print("=" * 60)

print(f"Total apples : {total}")
print(f"Raw          : {counts['raw']}")
print(f"Ripe         : {counts['ripe']}")
print(f"Rotten       : {counts['rotten']}")


# ============================================================
# ADD SUMMARY TO IMAGE
# ============================================================

summary = (
    f"Total: {total} | "
    f"Raw: {counts['raw']} | "
    f"Ripe: {counts['ripe']} | "
    f"Rotten: {counts['rotten']}"
)

cv2.rectangle(
    image,
    (10, 10),
    (850, 50),
    (0, 0, 0),
    -1
)

cv2.putText(
    image,
    summary,
    (20, 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE RESULT
# ============================================================

output_path = (
    OUTPUT_DIR /
    f"{INPUT_IMAGE.stem}_classified.jpg"
)

cv2.imwrite(
    str(output_path),
    image
)

print(f"\nResult saved to:")
print(output_path)

print("\nPipeline complete.")