import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\tkg91\PycharmProjects\Apple_Detection_Capstone"
)

YOLO_MODEL = PROJECT / "models" / "apple_detection_best.pt"

MATURITY_MODEL = (
    PROJECT
    / "models"
    / "orchard_maturity_resnet50_best.pth"
)

INPUT_IMAGE = (
    PROJECT
    / "test_images"
    / "20150919_174151_image1.png"
)

OUTPUT_DIR = PROJECT / "results" / "combined"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "orchard_maturity_pipeline.jpg"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "immature",
    "mature",
    "semi_mature"
]

IMAGE_SIZE = 224


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO11...")

detector = YOLO(
    str(YOLO_MODEL)
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("Loading orchard maturity ResNet50...")

classifier = models.resnet50(
    weights=None
)

classifier.fc = nn.Linear(
    classifier.fc.in_features,
    len(CLASS_NAMES)
)

classifier.load_state_dict(
    torch.load(
        MATURITY_MODEL,
        map_location=DEVICE
    )
)

classifier = classifier.to(DEVICE)
classifier.eval()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    INPUT_IMAGE
).convert("RGB")


# ============================================================
# YOLO DETECTION
# ============================================================

print("\nRunning apple detection...")

results = detector.predict(
    source=str(INPUT_IMAGE),
    imgsz=640,
    conf=0.25,
    verbose=False
)

result = results[0]

boxes = result.boxes.xyxy.cpu().numpy()

print(
    f"Apples detected: {len(boxes)}"
)


# ============================================================
# CLASSIFY EACH APPLE
# ============================================================

counts = {
    "immature": 0,
    "mature": 0,
    "semi_mature": 0
}

detections = []


for index, box in enumerate(boxes):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        continue

    crop = image.crop(
        (x1, y1, x2, y2)
    )

    input_tensor = transform(
        crop
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = classifier(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_index = prediction.item()

    class_name = CLASS_NAMES[class_index]

    class_confidence = confidence.item()

    counts[class_name] += 1

    detections.append({
        "box": (x1, y1, x2, y2),
        "class": class_name,
        "confidence": class_confidence
    })


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("ORCHARD MATURITY RESULTS")
print("========================================")

print(
    f"Total apples:      {len(detections)}"
)

print(
    f"Immature:          {counts['immature']}"
)

print(
    f"Semi-mature:       {counts['semi_mature']}"
)

print(
    f"Mature:            {counts['mature']}"
)


# ============================================================
# DRAW RESULTS
# ============================================================

from PIL import ImageDraw, ImageFont

output = image.copy()

draw = ImageDraw.Draw(output)


for detection in detections:

    x1, y1, x2, y2 = detection["box"]

    label = (
        f"{detection['class']} "
        f"{detection['confidence']:.2f}"
    )

    draw.rectangle(
        (x1, y1, x2, y2),
        outline="red",
        width=2
    )

    draw.text(
        (x1, max(0, y1 - 15)),
        label,
        fill="red"
    )


# ============================================================
# SAVE
# ============================================================

output.save(
    OUTPUT_IMAGE,
    quality=95
)

print("\nSaved:")
print(OUTPUT_IMAGE)