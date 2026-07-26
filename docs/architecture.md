# Architecture

This document describes the main technical choices behind Photos Quality.

## System Flow

```text
Uploaded image
    |
    v
FastAPI /predict endpoint
    |
    v
Pillow image loading and preprocessing
    |
    v
ResNet18 multi-label defect detection
    |
    v
Deterministic scoring layer
    |
    v
Groq suggestion generation
    |
    v
JSON response
```

## Backend

The backend is a FastAPI application. It exposes:

- `GET /health`
- `POST /predict`

The model is loaded once at startup from `MODEL_PATH`, which defaults to `data/model.pth`. If the file is missing, startup fails intentionally.

Uploaded images are written to a temporary file, analyzed, then deleted after the request finishes.

## Model

The model is based on ResNet18. The original ImageNet classification head is replaced with a linear layer that outputs one logit per defect label.

```text
Input image (224 x 224 x 3)
    |
    v
ResNet18 backbone
    |
    v
512-dimensional feature vector
    |
    v
Linear(512 -> 7)
    |
    v
Raw logits
```

The task is multi-label classification: a photo can have zero, one, or several quality issues at the same time.

## Transform Pipeline

Training and inference use the same basic image transform:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

The normalization values match the ImageNet statistics used by the pretrained ResNet18 backbone.

## Inference Pipeline

```text
PIL.Image.open()
    |
    v
convert("RGB")
    |
    v
Resize + ToTensor + Normalize
    |
    v
unsqueeze(0)
    |
    v
model(tensor) with torch.no_grad()
    |
    v
torch.sigmoid()
    |
    v
threshold at 0.5
    |
    v
issue dictionary
```

`torch.no_grad()` disables gradient tracking during inference, reducing memory usage and improving runtime performance.

## Scoring System

The quality score is not produced directly by the neural network. It is a deterministic business-rule layer built on top of detected issues and room type.

Base score: `100`

| Issue | Penalty |
|---|---:|
| `blurry` | -25 |
| `poor_space_visibility` | -25 |
| `bad_framing` | -15 |
| `low_light` | -15 |
| `cluttered` | -15 |
| `tilted` | -10 |
| `watermark` | -10 |

Room type adjustments:

| Room type | Adjustment |
|---|---:|
| `bathroom`, `attic` | +5 |
| `exterior`, `garden`, `pool` | -5 |

Grade scale:

| Score | Grade |
|---|---|
| 90-100 | A |
| 75-89 | B |
| 60-74 | C |
| 45-59 | D |
| 30-44 | E |
| 0-29 | F |

## LLM Suggestions

After scoring, detected issues are sent to Groq to generate concise French improvement suggestions.

The backend uses the OpenAI Python SDK with Groq's OpenAI-compatible API:

```python
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
```

If no issue is detected, the service returns a static message instead of making a network call.

## Why ResNet18

The dataset is small, so a larger model would be more likely to overfit. ResNet18 is lightweight, fast enough for API inference, and works well as a transfer learning baseline.

## Why Groq

Suggestions are generated synchronously during prediction, so latency matters. Groq provides a low-latency OpenAI-compatible API, which keeps the integration simple while allowing provider changes later.
