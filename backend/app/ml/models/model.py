import torch.nn as nn
from torchvision import models

NUM_OUTPUT_LABELS = 7


def get_model() -> nn.Module:
    """
    ResNet18 pré-entraîné sur ImageNet.
    La tête de classification native est remplacée par Linear(512 → 8)
    pour produire un logit par défaut détecté (multi-label).

    Le modèle retourne des logits bruts — le sigmoid est appliqué à l'inférence.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_OUTPUT_LABELS)
    return model
