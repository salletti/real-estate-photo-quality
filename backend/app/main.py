import os
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.ml.models.model import get_model

MODEL_PATH = os.getenv("MODEL_PATH", "data/model.pth")

app = FastAPI(title="Photos Quality", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()

if not Path(MODEL_PATH).exists():
    raise RuntimeError(f"Fine-tuned model weights not found at {MODEL_PATH}. Deployment aborted.")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
print(f"Fine-tuned weights loaded from {MODEL_PATH}")

app.state.model = model.to(device)
app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}
