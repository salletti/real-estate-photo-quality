import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from app.api.schemas import PredictResponse
from app.llm.llm_service import generate_human_suggestions
from app.ml.inference.predict import predict
from app.ml.scoring.scoring import compute_score

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
        request: Request,
        image: UploadFile,
        room_type: str = Form(default="other"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    model = request.app.state.model
    model.eval()

    try:
        ext = Path(image.filename).suffix.lower() if image.filename else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = Path(tmp.name)

        issues = predict(str(tmp_path), model)
        scoring = compute_score(issues, room_type)
        suggestions = generate_human_suggestions(issues)

        return {
            "issues": issues,
            "score": scoring["score"],
            "grade": scoring["grade"],
            "suggestions": suggestions,
        }

    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            os.remove(tmp_path)
