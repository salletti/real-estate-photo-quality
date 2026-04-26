from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABEL_COLUMNS = [
    "blurry",
    "low_light",
    "cluttered",
    "bad_framing",
    "tilted",
    "poor_space_visibility",
    "watermark",
]


class RealEstateDataset(Dataset):
    def __init__(self, images_dir: str, csv_path: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.datasetFile = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.datasetFile)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.datasetFile.iloc[idx]

        image_path = self.images_dir / row["image_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        labels = torch.tensor(
            [row[col] for col in LABEL_COLUMNS],
            dtype=torch.float32
        )

        return image, labels
