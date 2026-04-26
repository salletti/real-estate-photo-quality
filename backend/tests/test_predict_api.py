import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

STUB_ISSUES = {
    "blurry": 0,
    "low_light": 0,
    "cluttered": 0,
    "bad_framing": 0,
    "tilted": 0,
    "poor_space_visibility": 0,
    "watermark": 0,
}


@pytest.fixture
def client():
    app.state.model = MagicMock()
    with TestClient(app) as c:
        yield c


def _make_jpeg() -> io.BytesIO:
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_valid_image_returns_200(client):
    with patch("app.api.predict.predict", return_value=STUB_ISSUES), \
         patch("app.api.predict.generate_human_suggestions", return_value="Aucun problème."):
        response = client.post(
            "/predict",
            files={"image": ("photo.jpg", _make_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200


def test_predict_response_contains_required_fields(client):
    with patch("app.api.predict.predict", return_value=STUB_ISSUES), \
         patch("app.api.predict.generate_human_suggestions", return_value="Aucun problème."):
        response = client.post(
            "/predict",
            files={"image": ("photo.jpg", _make_jpeg(), "image/jpeg")},
        )
    body = response.json()
    assert "issues" in body
    assert "score" in body
    assert "grade" in body
    assert "suggestions" in body


def test_predict_non_image_file_returns_400(client):
    response = client.post(
        "/predict",
        files={"image": ("doc.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400
