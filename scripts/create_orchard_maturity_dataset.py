import json
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(
    r"C:\Users\tkg91\Downloads\A Multi-Stage, Pixel-Level Annotated Apple Dataset"
    r"\A Multi-Stage, Pixel-Level Annotated Apple Dataset"
    r"\dataset-20260508"
    r"\dataset-20260508"
)

OUTPUT = Path(
    r"C:\Users\tkg91\PycharmProjects\Apple_Detection_Capstone"
    r"\orchard_maturity"
)

# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "immature apple": "immature",
    "semi-mature apple": "semi_mature",
    "mature apple": "mature",
}

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for label in LABEL_MAP.values():
        (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

total = 0
skipped_unknown = 0
skipped_invalid = 0

for split in ["train", "val", "test"]:

    json_path = SOURCE / f"{split}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_root = SOURCE / split

    for image_id, item in data.items():

        filename = item.get("filename")

        if not filename:
            continue

        image_path = image_root / filename

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Could not open: {image_path}")
            continue

        width, height = image.size

        regions = item.get("regions", [])

        apple_number = 0

        for region in regions:

            region_attributes = region.get(
                "region_attributes", {}
            )

            original_label = region_attributes.get("name")

            # Ignore the single UNKNOWN annotation
            if original_label not in LABEL_MAP:
                skipped_unknown += 1
                continue

            label = LABEL_MAP[original_label]

            shape = region.get("shape_attributes", {})

            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])

            if not xs or not ys:
                skipped_invalid += 1
                continue

            # ------------------------------------------------
            # Bounding box around polygon
            # ------------------------------------------------

            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(width, int(max(xs)))
            y2 = min(height, int(max(ys)))

            if x2 <= x1 or y2 <= y1:
                skipped_invalid += 1
                continue

            crop = image.crop((x1, y1, x2, y2))

            # Ignore extremely tiny crops
            if crop.width < 10 or crop.height < 10:
                skipped_invalid += 1
                continue

            apple_number += 1

            output_name = (
                f"{Path(filename).stem}"
                f"_apple_{apple_number}.jpg"
            )

            output_path = (
                OUTPUT
                / split
                / label
                / output_name
            )

            crop.save(output_path, quality=95)

            total += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("ORCHARD MATURITY DATASET CREATED")
print("========================================")

print(f"Total apple crops: {total}")
print(f"Unknown annotations skipped: {skipped_unknown}")
print(f"Invalid/tiny annotations skipped: {skipped_invalid}")

print("\nClass distribution:")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    for label in LABEL_MAP.values():
        folder = OUTPUT / split / label
        count = len(list(folder.glob("*.jpg")))
        print(f"{label}: {count}")

print("\nOutput:")
print(OUTPUT)