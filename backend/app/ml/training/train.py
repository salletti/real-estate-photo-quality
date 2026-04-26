import torch
from pathlib import Path
from torch.utils.data import DataLoader

from app.ml.dataset.dataset import RealEstateDataset
from app.ml.dataset.transforms import get_transforms
from app.ml.models.model import get_model

IMAGES_DIR = "data/images"
CSV_PATH = "data/dataset.csv"
MODEL_PATH = "data/model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RealEstateDataset(IMAGES_DIR, CSV_PATH, transforms=get_transforms())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model().to(device)

    # BCEWithLogitsLoss : adapté au multi-label (chaque défaut est indépendant)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
