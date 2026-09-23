import cv2
import numpy as np
import os
import shutil
import random

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "minneapple",
    "detection",
    "train",
    "images"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "minneapple",
    "detection",
    "train",
    "masks"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

# ============================================================
# SETTINGS
# ============================================================

VAL_RATIO = 0.20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(
        os.path.join(OUTPUT_DIR, folder),
        exist_ok=True
    )

# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

image_files.sort()

print("Total images:", len(image_files))

# ============================================================
# SHUFFLE FOR TRAIN/VALIDATION SPLIT
# ============================================================

random.shuffle(image_files)

val_count = int(len(image_files) * VAL_RATIO)

val_files = image_files[:val_count]
train_files = image_files[val_count:]

print("Training images:", len(train_files))
print("Validation images:", len(val_files))

# ============================================================
# CONVERT ONE IMAGE + MASK
# ============================================================

def convert_image(filename, split):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image:", filename)
        return

    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print("ERROR: Could not read mask:", filename)
        return

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    image_height, image_width = image.shape[:2]

    mask_height, mask_width = mask.shape[:2]

    if (
        image_height != mask_height
        or image_width != mask_width
    ):
        print(
            "ERROR: Image/mask size mismatch:",
            filename
        )
        return

    # --------------------------------------------------------
    # Get individual object IDs
    # --------------------------------------------------------

    object_ids = np.unique(mask)

    # Remove background
    object_ids = object_ids[
        object_ids != 0
    ]

    # --------------------------------------------------------
    # YOLO annotation lines
    # --------------------------------------------------------

    yolo_lines = []

    for object_id in object_ids:

        # Create binary mask for this object
        object_mask = (
            mask == object_id
        ).astype(np.uint8)

        # Get pixel coordinates
        ys, xs = np.where(
            object_mask > 0
        )

        if len(xs) == 0:
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_min = xs.min()
        x_max = xs.max()

        y_min = ys.min()
        y_max = ys.max()

        # ----------------------------------------------------
        # Convert to YOLO format
        # ----------------------------------------------------

        box_width = x_max - x_min + 1
        box_height = y_max - y_min + 1

        x_center = (
            x_min + x_max
        ) / 2.0

        y_center = (
            y_min + y_max
        ) / 2.0

        # Normalize
        x_center /= image_width
        y_center /= image_height

        box_width /= image_width
        box_height /= image_height

        # Class 0 = apple
        line = (
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(line)

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    image_output = os.path.join(
        OUTPUT_DIR,
        "images",
        split,
        filename
    )

    label_filename = (
        os.path.splitext(filename)[0]
        + ".txt"
    )

    label_output = os.path.join(
        OUTPUT_DIR,
        "labels",
        split,
        label_filename
    )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        image_output
    )

    # --------------------------------------------------------
    # Write YOLO labels
    # --------------------------------------------------------

    with open(
        label_output,
        "w"
    ) as f:

        f.write(
            "\n".join(yolo_lines)
        )

    return len(yolo_lines)


# ============================================================
# PROCESS TRAINING IMAGES
# ============================================================

print("\nConverting training images...")

train_objects = 0

for i, filename in enumerate(train_files):

    count = convert_image(
        filename,
        "train"
    )

    if count is not None:
        train_objects += count

    if (i + 1) % 50 == 0:
        print(
            f"Processed {i + 1}/{len(train_files)}"
        )

# ============================================================
# PROCESS VALIDATION IMAGES
# ============================================================

print("\nConverting validation images...")

val_objects = 0

for i, filename in enumerate(val_files):

    count = convert_image(
        filename,
        "val"
    )

    if count is not None:
        val_objects += count

    if (i + 1) % 25 == 0:
        print(
            f"Processed {i + 1}/{len(val_files)}"
        )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("CONVERSION COMPLETE")

print("=" * 60)

print(
    "Training images:",
    len(train_files)
)

print(
    "Validation images:",
    len(val_files)
)

print(
    "Training apple instances:",
    train_objects
)

print(
    "Validation apple instances:",
    val_objects
)

print(
    "\nDataset created at:"
)

print(OUTPUT_DIR)