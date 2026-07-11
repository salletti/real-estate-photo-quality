from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from torchvision import models

from app.ml.models.model import get_model


def _mock_resnet() -> MagicMock:
    model = MagicMock()
    model.fc = SimpleNamespace(in_features=512)
    return model


def test_model_mode_training_uses_imagenet_weights(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "training")
    model = _mock_resnet()

    with patch("app.ml.models.model.models.resnet18", return_value=model) as resnet18:
        get_model()

    resnet18.assert_called_once_with(weights=models.ResNet18_Weights.DEFAULT)


def test_model_mode_inference_uses_no_initial_weights(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "inference")
    model = _mock_resnet()

    with patch("app.ml.models.model.models.resnet18", return_value=model) as resnet18:
        get_model()

    resnet18.assert_called_once_with(weights=None)


def test_model_mode_defaults_to_inference(monkeypatch):
    monkeypatch.delenv("MODEL_MODE", raising=False)
    model = _mock_resnet()

    with patch("app.ml.models.model.models.resnet18", return_value=model) as resnet18:
        get_model()

    resnet18.assert_called_once_with(weights=None)


def test_model_mode_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "TrAiNiNg")
    model = _mock_resnet()

    with patch("app.ml.models.model.models.resnet18", return_value=model) as resnet18:
        get_model()

    resnet18.assert_called_once_with(weights=models.ResNet18_Weights.DEFAULT)


def test_model_mode_invalid_value_raises_explicit_error(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "test")

    with pytest.raises(ValueError, match="Invalid MODEL_MODE: test. Expected: training or inference."):
        get_model()
