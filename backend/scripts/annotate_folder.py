import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from pillow_heif import register_avif_opener

register_avif_opener()  # active le support AVIF dans Pillow

SOURCE_DIR = Path("data/source_images")
IMAGES_DIR = Path("data/images")
CSV_PATH = Path("data/dataset.csv")

LABEL_COLUMNS = [
    "blurry",
    "low_light",
    "cluttered",
    "bad_framing",
    "tilted",
    "poor_space_visibility",
    "watermark",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".webp", ".avif"}


def build_row(image_name: str, room_type: str, issues: list[str]) -> dict:
    return {
        "image_name": image_name,
        "room_type": room_type,
        **{col: int(col in issues) for col in LABEL_COLUMNS},
    }


def annotate(issues: list[str], room_type: str, move: bool) -> None:
    images = [p for p in sorted(SOURCE_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print(f"No images found in {SOURCE_DIR}")
        return

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=["image_name", "room_type"] + LABEL_COLUMNS)

    if move:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for image in images:
        # Convertit les fichiers .avif en .jpg avant tout traitement
        if image.suffix.lower() == ".avif":
            jpg_path = image.with_suffix(".jpg")
            Image.open(image).convert("RGB").save(jpg_path, format="JPEG", quality=90)
            image.unlink()
            image = jpg_path

        row = build_row(image.name, room_type, issues)
        if image.name in df["image_name"].values:
            df.loc[df["image_name"] == image.name, list(row.keys())] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        if move:
            shutil.move(str(image), IMAGES_DIR / image.name)

    df.to_csv(CSV_PATH, index=False)

    action = f"annotated + moved to {IMAGES_DIR}" if move else "annotated"
    print(f"{len(images)} images {action} → {CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate images with quality issues")
    parser.add_argument("--issues", nargs="*", default=[], choices=LABEL_COLUMNS, metavar="ISSUE")
    parser.add_argument("--room_type", default="living_room")
    parser.add_argument("--move", action="store_true", help="Move images to data/images/")
    args = parser.parse_args()

    annotate(args.issues, args.room_type, args.move)


if __name__ == "__main__":
    main()
