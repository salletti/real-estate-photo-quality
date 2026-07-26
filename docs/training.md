# Training

This document explains how the model is trained and how model modes are used.

## Model Modes

`MODEL_MODE` controls the initial ResNet18 weights used when creating the model.

Training mode initializes the model for transfer learning:

```bash
MODEL_MODE=training python -m app.ml.training.train
```

Inference mode loads a fine-tuned model from disk:

```text
MODEL_MODE=inference
MODEL_PATH=data/model.pth
```

When `MODEL_MODE` is absent, the default is `inference`.

Production should not train a model or download ImageNet weights at startup. It should only load the already trained weights from `MODEL_PATH`.

## Running Training

Once the dataset is prepared:

```bash
cd backend
source .venv/bin/activate
MODEL_MODE=training python -m app.ml.training.train
```

With Docker:

```bash
docker compose run --rm backend python -m app.ml.training.train
```

## What The Script Does

1. Loads `data/images/` and `data/dataset.csv`
2. Initializes a ResNet18 model
3. Replaces the final fully connected layer with a new multi-label head
4. Trains for 5 epochs
5. Prints loss after each epoch

## Hyperparameters

Defined in `backend/app/ml/training/train.py`.

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |

## Multi-Label Classification

Each image can contain several defects at once. This makes the problem multi-label, not multi-class.

The model outputs one logit per issue. `BCEWithLogitsLoss` is used because each issue is treated as an independent binary prediction.

`CrossEntropyLoss` would be incorrect here because it assumes exactly one class is correct.

## Device Selection

The training script auto-detects CUDA:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No manual GPU configuration is required.

## Current Training Limitations

- No validation split
- No early stopping
- No checkpoint history
- No training-time data augmentation
- No threshold calibration

These are the highest-priority improvements before treating the model as production-grade.
