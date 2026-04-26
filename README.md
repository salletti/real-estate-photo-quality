# Photos Quality

Computer Vision API for real estate photo quality assessment.

A photo is uploaded, a ResNet18 model detects visual defects, a weighted scoring system computes a quality grade, and a Groq LLM generates human-readable improvement suggestions in French.

---

## Table of Contents

1. [Stack & Rationale](#1-stack--rationale)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running the API](#5-running-the-api)
6. [Running Tests](#6-running-tests)
7. [Training the Model](#7-training-the-model)
8. [Dataset Pipeline](#8-dataset-pipeline)
9. [ML Pipeline — Architecture & Design Choices](#9-ml-pipeline--architecture--design-choices)
   - [Model Architecture](#91-model-architecture)
   - [Multi-Label Classification](#92-multi-label-classification)
   - [Transform Pipeline](#93-transform-pipeline)
   - [Training Loop](#94-training-loop)
   - [Inference Pipeline](#95-inference-pipeline)
   - [Scoring System](#96-scoring-system)
10. [LLM Integration (Groq)](#10-llm-integration-groq)
11. [API Reference](#11-api-reference)
12. [Limitations & Known Issues](#12-limitations--known-issues)
13. [Future Improvements](#13-future-improvements)

---

## 1. Stack & Rationale

| Layer | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** | Native async, automatic OpenAPI docs, Pydantic validation out of the box |
| ML framework | **PyTorch 2.4 + TorchVision** | Flexible, mature ecosystem, native transfer learning support |
| Model | **ResNet18** | Lightweight, fast inference, strong ImageNet features, easy to fine-tune with small datasets |
| Image processing | **Pillow** | Standard PIL interface, consistent with TorchVision transforms |
| LLM | **Groq (llama-3.3-70b-versatile)** | Near-zero latency inference (~200ms), free tier, OpenAI-compatible API |
| Server | **Uvicorn** | ASGI, production-grade, native FastAPI integration |
| Containerization | **Docker + Compose** | Reproducible environment, easy GPU/CPU switching |

**Why ResNet18 and not a larger model?**

The dataset is small (a few hundred annotated images). A larger model like ResNet50 or EfficientNet would overfit immediately. ResNet18 has ~11M parameters, most of which are frozen from ImageNet pre-training. Only the classifier head is retrained — this makes the learning task tractable even with limited data.

**Why Groq and not OpenAI?**

Groq uses LPU (Language Processing Unit) hardware that delivers inference ~10× faster than GPU-based providers at equivalent price points. Since suggestions are generated synchronously per request, latency matters. The OpenAI-compatible API also means zero SDK change — we use the `openai` Python SDK pointed at `api.groq.com`.

---

## 2. Project Structure

```
photos-quality/
├── docker-compose.yml
├── README.md
├── .env
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env / .env.example
│   │
│   ├── app/
│   │   ├── main.py                  # FastAPI app, startup, model initialization
│   │   ├── api/
│   │   │   ├── predict.py           # POST /predict endpoint
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   ├── core/                    # Settings, config
│   │   ├── llm/
│   │   │   └── llm_service.py       # Groq API client, prompt construction
│   │   └── ml/
│   │       ├── dataset/
│   │       │   ├── dataset.py       # RealEstateDataset (torch.utils.data.Dataset)
│   │       │   └── transforms.py    # Image preprocessing pipeline
│   │       ├── models/
│   │       │   └── model.py         # ResNet18 with custom head
│   │       ├── training/
│   │       │   └── train.py         # Training loop
│   │       ├── inference/
│   │       │   └── predict.py       # Inference logic, threshold application
│   │       ├── scoring/
│   │       │   └── scoring.py       # Score & grade computation
│   │       └── suggestions/
│   │           └── suggestions.py   # Static fallback suggestions
│   │
│   ├── scripts/
│   │   ├── generate_dataset.py      # Synthetic dataset generation
│   │   ├── create_issue.py          # Apply a single defect to a batch of images
│   │   ├── annotate_folder.py       # Generate/update dataset.csv
│   │   └── image_transforms.py      # PIL-based defect simulation functions
│   │
│   ├── data/
│   │   ├── raw_images/              # Original source photos (unmodified)
│   │   ├── source_images/           # Processed intermediates
│   │   ├── images/                  # Final training images
│   │   ├── model.pth                # Trained model weights
│   │   └── dataset.csv               # Ground-truth annotations
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_predict_api.py
│       ├── test_scoring.py
│       └── test_suggestions.py
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── api/
        │   └── predict.js           # API client (POST /predict)
        └── components/
            ├── InfoSection.jsx
            ├── ResultCard.jsx       # Score, grade, suggestions display
            └── UploadForm.jsx       # Image upload + room type selector
```

---

## 3. Installation

### Option A — Docker (recommended)

```bash
# Clone the repo
git clone <repo-url>
cd photos-quality

# Copy and fill the environment file
cp backend/.env.example backend/.env
# Edit backend/.env and set GROQ_API_KEY

# Build and start
docker-compose up --build
```

The API will be available at `http://localhost:8000`.  
The frontend will be available at `http://localhost:5173`.

### Option B — Local Python environment

**Prerequisites:** Python 3.9+, pip, Node.js 18+

```bash
# --- Backend ---
cd backend

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# Start the API
uvicorn app.main:app --reload --port 8000

# --- Frontend (separate terminal) ---
cd frontend
npm install
npm run dev
```

---

## 4. Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key from [console.groq.com](https://console.groq.com) |

The application has **no database**. The trained model weights are loaded from disk at startup. All temporary files (uploaded images) are written to a temp directory and deleted after each request.

---

## 5. Running the API

```bash
# Development (with hot reload)
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1` when running with PyTorch. Multiple workers each load the model into memory independently — this is often acceptable on CPU but can exhaust VRAM on GPU.

**Verify the API is running:**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Interactive API docs:**

```
http://localhost:8000/docs
```

---

## 6. Running Tests

Tests are located in `backend/tests/` and use **pytest**.

```bash
cd backend
source .venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_scoring.py
pytest tests/test_suggestions.py
pytest tests/test_predict_api.py

# Run with coverage report
pytest --cov=app tests/
```

**Test files:**

| File | What it covers |
|---|---|
| `tests/test_scoring.py` | Score computation, grade assignment, room-type adjustments |
| `tests/test_suggestions.py` | Static fallback suggestions, issue-to-French mapping |
| `tests/test_predict_api.py` | `POST /predict` endpoint (FastAPI test client) |

---

## 7. Training the Model

Once the dataset is prepared (see [Dataset Pipeline](#8-dataset-pipeline)):

```bash
# From the project root
python -m app.ml.training.train
```

**What happens:**

1. The script loads `data/images/` and `data/dataset.csv`
2. It initializes a ResNet18 with ImageNet pre-trained weights
3. The final FC layer is replaced with a new layer of 8 outputs
4. The model trains for 5 epochs with Adam (lr=1e-3)
5. Loss is printed after each epoch

**Hyperparameters (defined in `app/ml/training/train.py`):**

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 5 | Small dataset, more epochs risk overfitting |
| Batch size | 16 | Fits comfortably in CPU RAM; increase to 32 on GPU |
| Learning rate | 1e-3 | Standard Adam default; works well for fine-tuning the head |
| Optimizer | Adam | Adaptive LR, stable convergence on small datasets |
| Loss | BCEWithLogitsLoss | Multi-label binary classification — each of the 8 labels is independent |

**To train on GPU** (if available), the script auto-detects CUDA:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No manual configuration needed.

---

## 8. Dataset Pipeline

The dataset is built from real estate photos that have been annotated with binary quality labels. There are two paths to build it: using real photos you annotate manually, or generating a synthetic dataset from a small set of source images.

### 8.1 Synthetic Dataset Generation

If you have a small set of clean source images, the scripts can generate a full dataset by programmatically applying defects:

```bash
# Step 1 — Apply a specific defect type to a folder of images
docker compose run --rm backend python scripts/create_issue.py

# Step 2 — Generate the full synthetic dataset (clean + 7 defect variants per image)
docker compose run --rm backend python scripts/generate_dataset.py

# Step 3 — Annotate the generated images and produce dataset.csv
docker compose run --rm backend python scripts/annotate_folder.py --room_type living_room --move
```

**What `generate_dataset.py` does:**

For each source image, it produces 8 versions:
- 1 original (no defect)
- 7 variants, each with one defect applied

This results in a balanced dataset where each class has equal representation. The `dataset.csv` is automatically generated with the correct binary labels.

### 8.2 Defect Simulation Functions (`scripts/image_transforms.py`)

Each function takes a PIL Image and returns a modified PIL Image:

| Function | Simulation technique | Label |
|---|---|---|
| `apply_blur` | Gaussian blur (radius=4) | `blurry` |
| `apply_low_light` | Brightness reduction to 35% via `ImageEnhance.Brightness` | `low_light` |
| `apply_tilt` | Rotate ±10° | `tilted` |
| `apply_bad_framing` | Crop 25% from left side, resize back to original | `bad_framing` |
| `apply_poor_space` | Crop 35% from all sides (zoom in), resize back | `poor_space_visibility` |
| `apply_cluttered` | Crop center 70%, resize back | `cluttered` |
These simulations are intentionally simple — they reproduce the _visual signature_ of each defect (soft edges, dark image, skewed horizon, tight crop) without requiring actual photography.

**Note:** `unappealing_composition` was removed from V1. Its synthetic simulation (crop 40% from left) was too similar to `bad_framing`, introducing noise rather than signal. Subjective aesthetic criteria will be addressed in V2 once the dataset reaches sufficient volume and diversity.

### 8.3 Label Format (`data/dataset.csv`)

```csv
image_name,room_type,blurry,low_light,cluttered,bad_framing,tilted,poor_space_visibility,watermark
abc123.jpg,living_room,0,0,1,0,0,0,0
def456.jpg,bedroom,1,1,0,0,0,0,0
```

- Each row is one image
- `room_type` is metadata used by the scoring system (not fed to the model)
- The 7 binary columns are the ground-truth labels
- Multiple labels can be 1 simultaneously (a photo can be both blurry and low_light)

### 8.4 Manual Annotation

Place your images in `backend/data/source_images/`, then run:

```bash
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type bedroom \
  --issues blurry low_light \
  --move
```

**Parameters:**

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--room_type` | No | `living_room` | Room type — one of the 8 valid values |
| `--issues` | No | _(none — clean photo)_ | Defects present, space-separated |
| `--move` | No | _(disabled)_ | Move images from `data/source_images/` to `data/images/` |

**Valid `--issues` values:**

```
blurry  low_light  cluttered  bad_framing  tilted
poor_space_visibility  watermark
```

**Valid `--room_type` values:**

```
living_room  bedroom  kitchen  bathroom
exterior  garden  pool  attic
```

**Examples:**

```bash
# Clean kitchen photos — no defects, no move
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type kitchen

# Dark and blurry bedroom photos — annotate and move to data/images/
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type bedroom \
  --issues blurry low_light \
  --move

# Exterior photos with bad framing
docker compose run --rm backend python scripts/annotate_folder.py \
  --room_type exterior \
  --issues bad_framing \
  --move
```

The script reads all `.jpg`, `.jpeg`, `.webp` files from `data/source_images/`, creates or updates `data/dataset.csv`, and optionally moves the images to `data/images/`. Existing rows are updated in place; new images are appended.

---

## 9. ML Pipeline — Architecture & Design Choices

### 9.1 Model Architecture

**File:** `backend/app/ml/models/model.py`

```
Input image (224×224×3)
        │
        ▼
ResNet18 backbone (pre-trained on ImageNet)
  ├─ Conv1 (7×7, 64 filters, stride 2)
  ├─ MaxPool
  ├─ Layer1 (2× BasicBlock, 64 channels)
  ├─ Layer2 (2× BasicBlock, 128 channels)
  ├─ Layer3 (2× BasicBlock, 256 channels)
  ├─ Layer4 (2× BasicBlock, 512 channels)
  └─ AdaptiveAvgPool → flatten → 512-dim feature vector
        │
        ▼
Linear(512 → 8)
        │
        ▼
8 raw logits (one per defect class)
```

The original ResNet18 classification head (`Linear(512 → 1000)`) is replaced with `Linear(512 → 8)`. This is the only layer with randomly initialized weights — all other layers start from ImageNet pre-training.

**Why not freeze the backbone?**

With our dataset size, fully fine-tuning all layers can work if the learning rate is small enough. Adam at 1e-3 applied to the randomly initialized head dominates gradient updates — the backbone receives proportionally small updates, acting as a soft freeze. Pragmatic tradeoff between simplicity and control.

### 9.2 Multi-Label Classification

**File:** `app/ml/dataset/dataset.py`, `app/ml/training/train.py`

Each image can have **zero or more** defects simultaneously. This is a multi-label problem, not multi-class.

**Loss function: `BCEWithLogitsLoss`**

This combines a sigmoid activation with binary cross-entropy in a numerically stable way:

```
loss = -[y · log(σ(x)) + (1 - y) · log(1 - σ(x))]
```

For each of the 8 outputs, the loss is computed independently. The total loss is the mean across all 8 outputs and all samples in the batch.

Using `BCEWithLogitsLoss` instead of `CrossEntropyLoss` is the key design decision here. `CrossEntropyLoss` assumes exactly one correct class — it would be wrong for this task.

### 9.3 Transform Pipeline

**File:** `app/ml/dataset/transforms.py`

```python
transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet18 expects 224×224
    transforms.ToTensor(),           # PIL [0, 255] → tensor [0.0, 1.0]
    transforms.Normalize(            # ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**Why these exact normalization values?**

These are the channel-wise mean and standard deviation computed across the entire ImageNet dataset. Applying them ensures that the input distribution seen at inference matches the distribution the backbone was originally trained on. Without this normalization, the pre-trained feature maps would be activated incorrectly and transfer learning would be significantly less effective.

The same transform is applied identically during training and inference — there is no separate augmentation pipeline for training in the current implementation. Adding augmentation (random flips, color jitter) would improve generalization and is the most impactful next step.

### 9.4 Training Loop

**File:** `app/ml/training/train.py`

```
for epoch in range(num_epochs):
    for batch in dataloader:
        images, labels = batch          # images: (B, 3, 224, 224), labels: (B, 8)
        optimizer.zero_grad()
        outputs = model(images)         # logits: (B, 8)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
```

The loop is intentionally minimal. There is no validation split, no early stopping, and no checkpoint saving in the current version. These are the most critical missing pieces before moving toward production.

**Standard gradient flow:**
1. `zero_grad()` — clear accumulated gradients from previous step
2. Forward pass — compute logits
3. `loss.backward()` — backpropagation, compute ∂loss/∂weights for all parameters
4. `optimizer.step()` — apply Adam update rule to all parameters

### 9.5 Inference Pipeline

**File:** `backend/app/ml/inference/predict.py`

```
Uploaded image file (bytes)
        │
        ▼
PIL.Image.open() → convert("RGB")      # Force 3-channel (handles RGBA, grayscale)
        │
        ▼
Apply transforms (Resize + ToTensor + Normalize)
        │                               # tensor shape: (3, 224, 224)
        ▼
unsqueeze(0)                            # add batch dim → (1, 3, 224, 224)
        │
        ▼
model(tensor)  [torch.no_grad()]        # logits: (1, 8)
        │
        ▼
torch.sigmoid()                         # probabilities: (1, 8) ∈ [0, 1]
        │
        ▼
squeeze(0)                              # (8,)
        │
        ▼
threshold at 0.5                        # binary: {label: 0 or 1}
        │
        ▼
{"blurry": 0, "low_light": 1, ...}
```

**`torch.no_grad()` context:**

Disables gradient computation during inference. This reduces memory usage by ~50% (no gradient buffers allocated) and increases forward-pass speed. It has no effect on the output values.

### 9.6 Scoring System

**File:** `app/ml/scoring/scoring.py`

The quality score is a deterministic function of the detected issues and the room type. It is not a model output — it is a business rule layer on top of the model predictions.

**Base score:** 100

**Issue penalties:**

| Issue | Penalty | Rationale |
|---|---|---|
| `blurry` | −25 | Most disqualifying defect in real estate photography |
| `poor_space_visibility` | −25 | Directly undermines the purpose of the photo |
| `bad_framing` | −15 | Significantly reduces visual appeal |
| `low_light` | −15 | Hard to recover in post-processing |
| `cluttered` | −15 | Distracts buyers, hard to visualize the space |
| `tilted` | −10 | Correctable in post-processing, less critical |
| `watermark` | −10 | Indicates use of a stock/agency photo |

**Room-type adjustments:**

| Room type | Adjustment | Rationale |
|---|---|---|
| `bathroom`, `attic` | +5 | Harder to photograph well; lower bar is expected |
| `exterior`, `garden`, `pool` | −5 | Higher visual expectations from buyers |

**Grade scale:**

| Score | Grade |
|---|---|
| 90–100 | A |
| 75–89 | B |
| 60–74 | C |
| 45–59 | D |
| 30–44 | E |
| 0–29 | F |

The penalties are intentionally additive. A photo with 4 issues will receive a very low score regardless of room type — this reflects that multiple simultaneous defects are not forgivable in professional real estate photography.

---

## 10. LLM Integration (Groq)

**File:** `app/llm/llm_service.py`

After the model predicts issues and the score is computed, the detected defect list is sent to Groq to generate actionable suggestions in French.

**Model:** `llama-3.3-70b-versatile`  
**Max tokens:** 300  
**Temperature:** 0.7

**Prompt construction:**

```
System:
  Tu es un expert en photographie immobilière.
  Génère des conseils concis, naturels et pratiques en français.
  Réponds en 3 à 5 phrases maximum.

User:
  Une photo immobilière présente les défauts suivants :
  - manque de luminosité
  - espace encombré

  Donne des conseils simples, concrets et naturels en français pour améliorer cette photo.
```

**Issue name mapping (English → French):**

| Model label | French label in prompt |
|---|---|
| blurry | flou |
| low_light | manque de luminosité |
| cluttered | espace encombré |
| bad_framing | mauvais cadrage |
| tilted | image penchée |
| poor_space_visibility | mauvaise visibilité de l'espace |
| watermark | filigrane visible |

**Fallback:** If no issues are detected (score near 100), the function returns a static string without calling the API: `"Cette photo ne présente aucun défaut détecté."` — this avoids a network call for clean images.

**Why use the OpenAI SDK for Groq?**

Groq exposes an OpenAI-compatible REST API. The `openai` Python SDK handles authentication, retries, and response parsing. The only change is the `base_url`:

```python
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
```

This makes a future provider swap (OpenAI, Mistral, etc.) a one-line change.

---

## 11. API Reference

### `POST /predict`

Analyze a real estate photo and return a quality score with suggestions.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | Yes | JPEG or PNG photo |
| `room_type` | string | Yes | One of: `living_room`, `bedroom`, `kitchen`, `bathroom`, `exterior`, `garden`, `pool`, `attic`, `other` |

**Response:** `application/json`

```json
{
  "issues": {
    "blurry": 0,
    "low_light": 1,
    "cluttered": 0,
    "bad_framing": 0,
    "tilted": 0,
    "poor_space_visibility": 0,
    "watermark": 0
  },
  "score": 85,
  "grade": "B",
  "suggestions": "La photo manque de luminosité. Essayez de prendre la photo en journée avec les volets ouverts pour maximiser la lumière naturelle. Un trépied peut aussi aider à éviter le flou de bougé en basse lumière."
}
```

**Errors:**

| Status | Cause |
|---|---|
| 422 | Missing or invalid field |
| 500 | Model inference failure or Groq API error |

### `GET /health`

```json
{"status": "ok"}
```

---

## 12. Limitations & Known Issues

> **The model's predictions should not be treated as ground truth.** The dataset is small, class imbalance has not been addressed, and the model has not been evaluated on a held-out test set. Scores and suggestions are indicative, not authoritative.

**Current limitations:**

- **Dataset size:** Training on a few hundred images is insufficient for robust generalization. A production-grade model would need thousands of diverse examples per class. The dataset remains experimental and requires enrichment.
- **No data augmentation:** The training pipeline applies no random augmentations (flips, color jitter, random crops). This increases overfitting risk on small datasets.
- **No validation split:** There is no held-out validation set and no early stopping. Overfitting cannot be detected during training.
- **Synthetic defects:** The dataset generation pipeline applies artificial defects (blur, brightness reduction, crops) to simulate real issues. Real-world photos exhibit these issues with much greater variability and co-occurrence patterns.
- **Single-label defect simulation:** The synthetic generation applies exactly one defect per image variant. Real photos frequently exhibit multiple simultaneous issues, which the model has limited training signal for.
- **Threshold is fixed:** The 0.5 inference threshold is a standard starting point. It has not been calibrated against a validation set using precision-recall curves.

---

## 13. Future Improvements

This section documents architectural improvements that were designed and validated but deliberately deferred — the dataset needs to grow before activating them would be net-positive.

### 13.1 Room Type Contextual Modeling

**The idea:** inject the room type as structured metadata into the model so that defect detection is context-aware. The same visual signal does not carry the same weight depending on the space:

- A **cluttered kitchen** is not judged like a **dense garden** — expected object density differs.
- A **dark bathroom** is not held to the same lighting standard as an **exterior shot**.
- An **attic** is inherently harder to photograph well than a **living room**.

**Proposed architecture:**

```
Image (3×224×224)       Room type (one-hot, 8)
        │                         │
        ▼                         │
ResNet18 backbone                 │
  └─ AdaptiveAvgPool → flatten    │
        │                         │
        ▼                         ▼
   512 visual features ─── cat ─── 8 room features
                              │
                              ▼
                        520 combined features
                              │
                              ▼
                  Linear(520 → 256) + ReLU + Dropout(0.3)
                              │
                              ▼
                        Linear(256 → 8) → logits
```

**Why it is deferred:**

The architecture was prototyped and the code infrastructure exists (all components were built). The decision to revert was data-centric, not technical:

- The current dataset does not have sufficient per-room diversity. A model trained with this fusion on a small, unbalanced dataset risks learning spurious correlations rather than genuine context.
- Adding architectural complexity without the data to support it degrades the baseline rather than improving it.
- The priority is to build a robust single-stream baseline first, then re-enable contextual fusion once the dataset reaches a critical mass per room type.

**Activation condition:** balanced dataset with ≥500 annotated images per room type across varied real-world conditions.

### 13.2 Automatic Room Type Detection

Rather than requiring the user to select the room type manually, a secondary classifier could predict it directly from the image. This would remove the only remaining user input from the form and make the pipeline fully automatic.

**Approach:** fine-tune a lightweight classifier (MobileNetV3 or EfficientNet-B0) on a labeled room-type dataset (e.g. MIT Indoor Scenes), then chain it as a pre-processing step before the quality model.

### 13.3 Data Augmentation

Adding random augmentations during training (horizontal flips, color jitter, random crops) would significantly reduce overfitting on the current small dataset with minimal code change.

### 13.4 Validation Split & Early Stopping

Splitting the dataset into train/val would allow monitoring generalization during training and stopping before the model overfits — the most impactful next step for training reliability.
