import json
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\tkg91\PycharmProjects\Apple_Detection_Capstone"
)

DATASET = Path(
    r"C:\Users\tkg91\Downloads"
    r"\A Multi-Stage, Pixel-Level Annotated Apple Dataset"
    r"\A Multi-Stage, Pixel-Level Annotated Apple Dataset"
    r"\dataset-20260508"
    r"\dataset-20260508"
)

MODEL_PATH = PROJECT / "models" / "apple_detection_best.pt"

OUTPUT = (
    PROJECT
    / "results"
    / "combined"
    / "orchard_detection_debug.jpg"
)


# ============================================================
# LOAD TEST JSON
# ============================================================

with open(
    DATASET / "test.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# TAKE FIRST IMAGE
# ============================================================

item = list(data.values())[0]

filename = item["filename"]

image_path = DATASET / "test" / filename

print("Image:", filename)
print("Path:", image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load {image_path}"
    )


# ============================================================
# DRAW GROUND TRUTH
# ============================================================

ground_truth_count = 0

for region in item.get("regions", []):

    label = region.get(
        "region_attributes",
        {}
    ).get("name")

    if label not in [
        "immature apple",
        "semi-mature apple",
        "mature apple"
    ]:
        continue

    shape = region.get(
        "shape_attributes",
        {}
    )

    xs = shape.get("all_points_x", [])
    ys = shape.get("all_points_y", [])

    if not xs or not ys:
        continue

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Short label
    short_label = {
        "immature apple": "I",
        "semi-mature apple": "S",
        "mature apple": "M"
    }[label]

    cv2.putText(
        image,
        short_label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    ground_truth_count += 1


# ============================================================
# YOLO
# ============================================================

model = YOLO(
    str(MODEL_PATH)
)

results = model.predict(
    source=str(image_path),
    imgsz=640,
    conf=0.25,
    verbose=False
)

result = results[0]

prediction_count = 0

if result.boxes is not None:

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    for box, confidence in zip(
        boxes,
        confidences
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            f"Y {confidence:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        prediction_count += 1


# ============================================================
# SUMMARY TEXT
# ============================================================

cv2.putText(
    image,
    f"Green=Ground Truth: {ground_truth_count}",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"Red=YOLO: {prediction_count}",
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(OUTPUT),
    image
)

print("\nGround truth:", ground_truth_count)
print("YOLO predictions:", prediction_count)

print("\nSaved:")
print(OUTPUT)