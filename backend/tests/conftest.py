from unittest.mock import MagicMock, patch

# Patch get_model before app.main is imported by test modules so that
# ResNet18 ImageNet weights are never downloaded during test collection.
_model_patch = patch("app.ml.models.model.get_model", return_value=MagicMock())
_model_patch.start()
