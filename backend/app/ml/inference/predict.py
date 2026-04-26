from pathlib import Path

import torch
from PIL import Image

from app.ml.dataset.dataset import LABEL_COLUMNS
from app.ml.dataset.transforms import get_transforms

THRESHOLD = 0.7


def predict(image_path: str, model: torch.nn.Module) -> dict[str, int]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    device = next(model.parameters()).device

    image = Image.open(path).convert("RGB")

    # (3, 224, 224) → unsqueeze → (1, 3, 224, 224)
    processed_image = get_transforms()(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(processed_image)  # (1, 8)

    probs = torch.sigmoid(logits).squeeze(0)  # (8,)

    return {
        label: int(probs[i].item() >= THRESHOLD)
        for i, label in enumerate(LABEL_COLUMNS)
    }
