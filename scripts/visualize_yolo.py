import cv2
import os

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "images",
    "train"
)

LABEL_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "labels",
    "train"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "visualized"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND AN IMAGE THAT EXISTS
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(image_files) == 0:
    print("ERROR: No images found in:")
    print(IMAGE_DIR)
    exit()

# Pick the first image
filename = sorted(image_files)[0]

print("Selected image:", filename)


# ============================================================
# PATHS
# ============================================================

image_path = os.path.join(
    IMAGE_DIR,
    filename
)

label_filename = (
    os.path.splitext(filename)[0]
    + ".txt"
)

label_path = os.path.join(
    LABEL_DIR,
    label_filename
)


# ============================================================
# CHECK LABEL
# ============================================================

if not os.path.exists(label_path):
    print("ERROR: Matching label not found:")
    print(label_path)
    exit()


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read image:")
    print(image_path)
    exit()

height, width = image.shape[:2]

print("Image width:", width)
print("Image height:", height)


# ============================================================
# READ LABELS
# ============================================================

with open(label_path, "r") as f:
    lines = f.readlines()

print("Number of annotations:", len(lines))


# ============================================================
# DRAW BOXES
# ============================================================

for line in lines:

    parts = line.strip().split()

    if len(parts) != 5:
        continue

    class_id = int(parts[0])

    x_center = float(parts[1])
    y_center = float(parts[2])
    box_width = float(parts[3])
    box_height = float(parts[4])

    # Convert normalized coordinates
    # to pixel coordinates

    x_center *= width
    y_center *= height

    box_width *= width
    box_height *= height

    x1 = int(
        x_center - box_width / 2
    )

    y1 = int(
        y_center - box_height / 2
    )

    x2 = int(
        x_center + box_width / 2
    )

    y2 = int(
        y_center + box_height / 2
    )

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw class name
    cv2.putText(
        image,
        "apple",
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


# ============================================================
# SAVE
# ============================================================

output_filename = (
    os.path.splitext(filename)[0]
    + "_verified.jpg"
)

output_path = os.path.join(
    OUTPUT_DIR,
    output_filename
)

cv2.imwrite(
    output_path,
    image
)

print("\nVerification image saved:")
print(output_path)