import json
from pathlib import Path

import cv2
import numpy as np
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

# ============================================================
# SETTINGS
# ============================================================

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640


# ============================================================
# LOAD DATA
# ============================================================

with open(DATASET / "test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

model = YOLO(str(MODEL_PATH))


# ============================================================
# FUNCTIONS
# ============================================================

def polygon_to_box(region):
    """
    Convert polygon annotation into bounding box.
    """

    shape = region.get("shape_attributes", {})

    xs = shape.get("all_points_x", [])
    ys = shape.get("all_points_y", [])

    if not xs or not ys:
        return None

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    ]


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# EVALUATION
# ============================================================

total_ground_truth = 0
total_predictions = 0
total_true_positive = 0

image_results = []

for index, item in enumerate(test_data.values(), start=1):

    filename = item.get("filename")

    image_path = DATASET / "test" / filename

    if not image_path.exists():
        continue

    # --------------------------------------------------------
    # Ground truth boxes
    # --------------------------------------------------------

    ground_truth_boxes = []

    for region in item.get("regions", []):

        label = region.get(
            "region_attributes",
            {}
        ).get("name")

        # Ignore the single UNKNOWN annotation
        if label not in [
            "immature apple",
            "semi-mature apple",
            "mature apple"
        ]:
            continue

        box = polygon_to_box(region)

        if box is not None:
            ground_truth_boxes.append(box)

    # --------------------------------------------------------
    # YOLO predictions
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]

    prediction_boxes = []

    if result.boxes is not None:

        prediction_boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
            .tolist()
        )

    total_ground_truth += len(
        ground_truth_boxes
    )

    total_predictions += len(
        prediction_boxes
    )

    # --------------------------------------------------------
    # Match predictions to ground truth
    # --------------------------------------------------------

    matched_gt = set()

    true_positive = 0

    for prediction in prediction_boxes:

        best_iou = 0
        best_gt_index = None

        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                prediction,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            true_positive += 1
            matched_gt.add(best_gt_index)

    total_true_positive += true_positive

    image_results.append({
        "filename": filename,
        "ground_truth": len(ground_truth_boxes),
        "predictions": len(prediction_boxes),
        "true_positive": true_positive
    })

    print(
        f"[{index}/{len(test_data)}] "
        f"{filename}: "
        f"GT={len(ground_truth_boxes)}, "
        f"Pred={len(prediction_boxes)}, "
        f"TP={true_positive}"
    )


# ============================================================
# METRICS
# ============================================================

false_positive = (
    total_predictions -
    total_true_positive
)

false_negative = (
    total_ground_truth -
    total_true_positive
)

precision = (
    total_true_positive /
    total_predictions
    if total_predictions > 0
    else 0
)

recall = (
    total_true_positive /
    total_ground_truth
    if total_ground_truth > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("ORCHARD YOLO DETECTION RESULTS")
print("========================================")

print(
    f"Images evaluated:     {len(image_results)}"
)

print(
    f"Ground truth apples:   {total_ground_truth}"
)

print(
    f"Predicted apples:      {total_predictions}"
)

print(
    f"True positives:        {total_true_positive}"
)

print(
    f"False positives:       {false_positive}"
)

print(
    f"False negatives:       {false_negative}"
)

print(
    f"Precision @ IoU 0.50:  {precision:.4f}"
)

print(
    f"Recall @ IoU 0.50:     {recall:.4f}"
)

print(
    f"F1 @ IoU 0.50:         {f1:.4f}"
)