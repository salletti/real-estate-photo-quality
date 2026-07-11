import os

import torch.nn as nn
from torchvision import models

NUM_OUTPUT_LABELS = 7
MODEL_MODE_TRAINING = "training"
MODEL_MODE_INFERENCE = "inference"


def get_model_mode() -> str:
    model_mode = os.getenv("MODEL_MODE", MODEL_MODE_INFERENCE).strip().lower()
    if model_mode not in {MODEL_MODE_TRAINING, MODEL_MODE_INFERENCE}:
        raise ValueError(
            f"Invalid MODEL_MODE: {model_mode}. "
            "Expected: training or inference."
        )
    return model_mode


def get_model() -> nn.Module:
    """
    ResNet18 avec poids ImageNet uniquement en mode entraînement.
    La tête de classification native est remplacée par Linear(512 → 7)
    pour produire un logit par défaut détecté (multi-label).

    Le modèle retourne des logits bruts — le sigmoid est appliqué à l'inférence.
    """
    weights = (
        models.ResNet18_Weights.DEFAULT
        if get_model_mode() == MODEL_MODE_TRAINING
        else None
    )
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_OUTPUT_LABELS)
    return model
