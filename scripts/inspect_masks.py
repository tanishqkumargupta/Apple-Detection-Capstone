import cv2
import numpy as np
import os

# Get the project root directory
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

print("Project root:", PROJECT_ROOT)
print("Image directory:", IMAGE_DIR)
print("Mask directory:", MASK_DIR)

# Check directories
if not os.path.exists(IMAGE_DIR):
    print("\nERROR: Image directory does not exist!")
    exit()

if not os.path.exists(MASK_DIR):
    print("\nERROR: Mask directory does not exist!")
    exit()

files = os.listdir(IMAGE_DIR)

print("\nNumber of images:", len(files))

for filename in files[:5]:

    image_path = os.path.join(IMAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not read image:", filename)
        continue

    if mask is None:
        print("Could not read mask:", filename)
        continue

    unique_values = np.unique(mask)

    print("\nFile:", filename)
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Number of unique mask values:", len(unique_values))
    print("Mask values:", unique_values[:100])