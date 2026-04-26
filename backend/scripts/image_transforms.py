import random

import numpy as np
from PIL import Image, ImageFilter


def apply_blurry(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=4))


def apply_low_light(image: Image.Image) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    arr = np.clip(arr * 0.35, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_tilted(image: Image.Image) -> Image.Image:
    angle = random.choice([-10, 10])
    return image.rotate(angle, expand=False)


def apply_bad_framing(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((int(w * 0.25), 0, w, h)).resize((w, h))


def apply_poor_space_visibility(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((int(w * 0.35), int(h * 0.35), w, h)).resize((w, h))


def apply_cluttered(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.7), int(h * 0.7))).resize((w, h))


TRANSFORMS: dict[str, callable] = {
    "blurry": apply_blurry,
    "low_light": apply_low_light,
    "tilted": apply_tilted,
    "bad_framing": apply_bad_framing,
    "poor_space_visibility": apply_poor_space_visibility,
    "cluttered": apply_cluttered,
}
