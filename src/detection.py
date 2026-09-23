from pathlib import Path
from ultralytics import YOLO
import cv2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "apple_detection_best.pt"
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "results" / "detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")


# ============================================================
# DETECTION
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    image for image in INPUT_DIR.iterdir()
    if image.suffix.lower() in image_extensions
]

if not images:
    print("No test images found.")
    exit()


total_images = 0
total_apples = 0


for image_path in images:

    print("\n" + "=" * 60)
    print(f"Processing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        continue

    # Run YOLO detection
    results = model(
        image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Number of detected apples
    if result.boxes is not None:
        apple_count = len(result.boxes)
    else:
        apple_count = 0

    print(f"Detected apples: {apple_count}")

    total_images += 1
    total_apples += apple_count

    # Draw bounding boxes
    annotated_image = result.plot()

    output_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"

    cv2.imwrite(
        str(output_path),
        annotated_image
    )

    print(f"Saved: {output_path}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DETECTION SUMMARY")
print("=" * 60)

print(f"Images processed : {total_images}")
print(f"Total detections : {total_apples}")
print(f"Results saved to : {OUTPUT_DIR}")