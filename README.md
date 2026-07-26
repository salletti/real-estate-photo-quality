# Photos Quality

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-f55036)
![License](https://img.shields.io/badge/License-MIT-yellow)

Computer vision application for assessing the quality of real estate photos.

Users upload a property photo, the backend detects visual defects with a fine-tuned ResNet18 model, computes a quality score, and generates practical improvement suggestions in French with Groq.

## Overview

Photos Quality combines a React frontend, a FastAPI backend, and a PyTorch image model. The application focuses on common real estate photography issues such as blur, low light, poor framing, tilted images, clutter, poor space visibility, and visible watermarks.

The ML output is converted into a deterministic score and grade, then enriched with human-readable advice for improving the photo.

## Features

- Upload a real estate photo from a web interface
- Detect multiple visual quality issues in the same image
- Compute a score from 0 to 100 and a grade from A to F
- Generate concise French improvement suggestions
- Expose predictions through a FastAPI API
- Run locally with Docker Compose

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite |
| Backend | FastAPI, Uvicorn |
| ML | PyTorch 2.4, TorchVision, ResNet18 |
| Image processing | Pillow |
| LLM suggestions | Groq via the OpenAI-compatible API |
| Runtime | Docker, Docker Compose |

## Quick Start

### Docker

```bash
git clone <repo-url>
cd photos-quality

cp backend/.env.example backend/.env
# Edit backend/.env and set GROQ_API_KEY

docker compose up --build
```

The frontend will be available at:

```text
http://localhost:5173
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Local Development

Prerequisites:

- Python 3.12
- Node.js 18+
- pip

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GROQ_API_KEY

uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Configuration

Backend configuration is read from environment variables. Start from `backend/.env.example`:

```bash
cp backend/.env.example backend/.env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | None | Groq API key used to generate suggestions |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS origins |
| `MODEL_MODE` | No | `inference` | Model initialization mode |
| `MODEL_PATH` | No | `data/model.pth` | Path to fine-tuned model weights |

The application has no database. The backend loads model weights from disk at startup and deletes uploaded temporary files after each request.

## API Usage

### `POST /predict`

Analyze a real estate photo and return detected issues, score, grade, and suggestions.

Request type: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | Yes | Image file |
| `room_type` | string | No | Room type used by the scoring layer. Defaults to `other`. |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@photo.jpg" \
  -F "room_type=living_room"
```

Response:

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
  "suggestions": "La photo manque de luminosité. Essayez de prendre la photo en journée avec les volets ouverts pour maximiser la lumière naturelle."
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

## Development

Run backend tests with pytest:

```bash
cd backend
source .venv/bin/activate
pytest
```

Useful test targets:

```bash
pytest tests/test_scoring.py
pytest tests/test_suggestions.py
pytest tests/test_predict_api.py
pytest tests/test_model_mode.py
```

## Documentation

- [Architecture](docs/architecture.md)
- [Training](docs/training.md)
- [Dataset pipeline](docs/dataset.md)
- [Production deployment](docs/deployment.md)

## Limitations

This project is experimental. The model is trained on a small dataset and should not be treated as a production-grade quality evaluator.

Current limitations:

- No validation split or held-out evaluation set
- No calibrated inference threshold
- Synthetic defects are simpler than real-world photo defects
- The scoring layer is deterministic and opinionated
- LLM suggestions depend on Groq availability

## Roadmap

- Add validation split and early stopping
- Add data augmentation during training
- Calibrate thresholds with precision-recall metrics
- Expand the dataset with more real annotated photos
- Add automatic room type detection
- Revisit room-aware model architecture after the dataset grows

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
