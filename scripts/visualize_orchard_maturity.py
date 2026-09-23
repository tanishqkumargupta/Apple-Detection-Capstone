from pathlib import Path
import random
from PIL import Image, ImageDraw

PROJECT = Path(r"C:\Users\tkg91\PycharmProjects\Apple_Detection_Capstone")
DATASET = PROJECT / "orchard_maturity"
OUTPUT = PROJECT / "results" / "classification" / "orchard_maturity_samples"

OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

for class_name in ["immature", "semi_mature", "mature"]:

    source = DATASET / "train" / class_name

    images = list(source.glob("*.jpg"))

    samples = random.sample(images, min(10, len(images)))

    for i, image_path in enumerate(samples):

        image = Image.open(image_path).convert("RGB")

        # Resize for easier viewing
        image.thumbnail((400, 400))

        draw = ImageDraw.Draw(image)

        draw.rectangle(
            [0, 0, image.width - 1, image.height - 1],
            outline="red",
            width=3
        )

        draw.text(
            (10, 10),
            class_name,
            fill="red"
        )

        output_path = OUTPUT / f"{class_name}_{i+1}.jpg"
        image.save(output_path, quality=95)

print("Visual samples created:")
print(OUTPUT)