import cv2
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def analyze_mask(mask):
    """
    Analyze one MinneApple instance mask.
    """

    values = np.unique(mask)

    # Remove background
    object_ids = values[values != 0]

    print("Number of object IDs:", len(object_ids))

    objects = []

    for object_id in object_ids:

        # Pixels belonging to this particular object
        object_mask = (mask == object_id).astype(np.uint8)

        # Find coordinates
        ys, xs = np.where(object_mask > 0)

        if len(xs) == 0:
            continue

        x_min = xs.min()
        y_min = ys.min()
        x_max = xs.max()
        y_max = ys.max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = len(xs)

        objects.append({
            "id": int(object_id),
            "area": area,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": width,
            "height": height
        })

    return objects


# Get image files
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print("Total training images:", len(image_files))

# Analyze first 5 images
for filename in image_files[:5]:

    print("\n" + "=" * 60)
    print("FILE:", filename)

    image_path = os.path.join(IMAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    # Check mask exists
    if not os.path.exists(mask_path):
        print("ERROR: Matching mask not found!")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: Could not read image")
        continue

    if mask is None:
        print("ERROR: Could not read mask")
        continue

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    # Analyze mask
    objects = analyze_mask(mask)

    print("Number of detected objects:", len(objects))

    # Print first 10 objects
    print("\nFirst 10 objects:")

    for obj in objects[:10]:

        print(
            f"ID={obj['id']} | "
            f"Area={obj['area']} | "
            f"BBox=({obj['x_min']}, {obj['y_min']}, "
            f"{obj['x_max']}, {obj['y_max']}) | "
            f"Size={obj['width']}x{obj['height']}"
        )