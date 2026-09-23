import os
import shutil
import random

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = "fruit_ripeness_dataset/dataset"
OUTPUT = "apple_ripeness"

RANDOM_SEED = 42
VAL_RATIO = 0.20

random.seed(RANDOM_SEED)

# Source folder -> final class name
CLASS_MAPPING = {
    "unripe apple": "raw",
    "freshapples": "ripe",
    "rottenapples": "rotten"
}

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in ["raw", "ripe", "rotten"]:
        os.makedirs(
            os.path.join(OUTPUT, split, class_name),
            exist_ok=True
        )

# ============================================================
# PROCESS DATA
# ============================================================

for source_class, final_class in CLASS_MAPPING.items():

    train_source = os.path.join(
        SOURCE, "train", source_class
    )

    test_source = os.path.join(
        SOURCE, "test", source_class
    )

    # --------------------------------------------------------
    # Get training images
    # --------------------------------------------------------

    train_files = [
        f for f in os.listdir(train_source)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    random.shuffle(train_files)

    # 80% train / 20% validation
    val_count = int(len(train_files) * VAL_RATIO)

    val_files = train_files[:val_count]
    train_files_final = train_files[val_count:]

    # --------------------------------------------------------
    # Copy training images
    # --------------------------------------------------------

    for filename in train_files_final:

        src = os.path.join(
            train_source, filename
        )

        dst = os.path.join(
            OUTPUT, "train", final_class, filename
        )

        shutil.copy2(src, dst)

    # --------------------------------------------------------
    # Copy validation images
    # --------------------------------------------------------

    for filename in val_files:

        src = os.path.join(
            train_source, filename
        )

        dst = os.path.join(
            OUTPUT, "val", final_class, filename
        )

        shutil.copy2(src, dst)

    # --------------------------------------------------------
    # Copy original test images
    # --------------------------------------------------------

    test_files = [
        f for f in os.listdir(test_source)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    for filename in test_files:

        src = os.path.join(
            test_source, filename
        )

        dst = os.path.join(
            OUTPUT, "test", final_class, filename
        )

        shutil.copy2(src, dst)

    print(
        f"{source_class:15} -> {final_class:7} | "
        f"Train: {len(train_files_final):4} | "
        f"Val: {len(val_files):4} | "
        f"Test: {len(test_files):4}"
    )

print("\n============================================")
print("APPLE RIPENESS DATASET CREATED")
print("============================================")
print(f"Location: {os.path.abspath(OUTPUT)}")