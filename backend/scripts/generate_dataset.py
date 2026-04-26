from pathlib import Path

import pandas as pd
from PIL import Image

from image_transforms import TRANSFORMS

RAW_DIR = Path("data/raw_images")
OUT_DIR = Path("data/generated")
IMAGES_DIR = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "dataset.csv"

ROOM_TYPE = "living_room"

LABEL_COLUMNS = [
    "blurry",
    "low_light",
    "cluttered",
    "bad_framing",
    "tilted",
    "poor_space_visibility",
    "watermark",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".webp"}


def empty_labels() -> dict:
    return {col: 0 for col in LABEL_COLUMNS}


def save_image(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=90)


def generate() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    raw_images = [p for p in sorted(RAW_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not raw_images:
        print(f"No images found in {RAW_DIR}")
        return

    rows = []
    counter = 1

    for source in raw_images:
        image = Image.open(source).convert("RGB")

        # original
        name = f"img_{counter:04d}.jpg"
        save_image(image, IMAGES_DIR / name)
        rows.append({"image_name": name, "room_type": ROOM_TYPE, **empty_labels()})
        counter += 1

        # variants
        for label, fn in TRANSFORMS.items():
            name = f"img_{counter:04d}.jpg"
            save_image(fn(image), IMAGES_DIR / name)
            labels = empty_labels()
            labels[label] = 1
            rows.append({"image_name": name, "room_type": ROOM_TYPE, **labels})
            counter += 1

    pd.DataFrame(rows, columns=["image_name", "room_type"] + LABEL_COLUMNS).to_csv(CSV_PATH, index=False)
    print(f"Generated {counter - 1} images → {CSV_PATH}")


if __name__ == "__main__":
    generate()
