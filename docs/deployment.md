# Production Deployment

Production is intended to run with Docker Compose behind a reverse proxy such as Coolify and Traefik.

The production stack is defined in `docker-compose.prod.yml`.

## Services

The stack contains two services:

- `frontend`: public HTTP service
- `backend`: private API service exposed only inside the Docker network

The backend exposes port `8000` to the private network but does not publish it to the host.

## Environment Variables

Configure production environment variables in the deployment platform, not in a committed `.env` file.

Required:

```text
GROQ_API_KEY
MODEL_MODE=inference
MODEL_PATH=data/model.pth
```

Optional:

```text
ALLOWED_ORIGINS
```

## Build And Run

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

## Frontend Proxy

The frontend is expected to proxy API calls:

```text
GET /api/health
POST /api/predict
```

The backend receives those requests as:

```text
GET /health
POST /predict
```

## Model Handling

The model is trained locally and production only performs inference.

The backend loads fine-tuned weights from:

```text
backend/data/model.pth
```

Inside the container, this is available as:

```text
data/model.pth
```

Production should not train a model, modify `model.pth`, or download TorchVision/ImageNet weights at startup.

## Health Checks

The backend health check calls:

```text
http://localhost:8000/health
```

The frontend health check calls:

```text
http://localhost/
```
