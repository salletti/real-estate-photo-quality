import argparse
import shutil
from pathlib import Path

from PIL import Image
from pillow_heif import register_avif_opener

register_avif_opener()  # active le support AVIF dans Pillow

from image_transforms import TRANSFORMS

RAW_DIR = Path("data/raw_images")
SOURCE_DIR = Path("data/source_images")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".webp", ".avif"}


def process(issue: str) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(RAW_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print(f"No images found in {RAW_DIR}")
        return

    transform = TRANSFORMS.get(issue)

    for source in images:
        dest = SOURCE_DIR / (source.stem + ".jpg")

        if transform:
            image = Image.open(source).convert("RGB")
            image = transform(image)
            image.save(dest, format="JPEG", quality=90)
        elif source.suffix.lower() == ".avif":
            Image.open(source).convert("RGB").save(dest, format="JPEG", quality=90)
        else:
            shutil.copy(source, dest)

        source.unlink()

    print(f"Processed {len(images)} images ({issue or 'no issue'}) → {SOURCE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an issue to images and move them to source_images")
    parser.add_argument("--issue", choices=list(TRANSFORMS.keys()), default=None)
    args = parser.parse_args()

    process(args.issue)


if __name__ == "__main__":
    main()
