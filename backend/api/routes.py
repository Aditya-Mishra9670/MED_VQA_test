"""
API route definitions for the Medical VQA System.

Endpoints:
    POST /predict       — Main VQA prediction (image + question → answer + visuals)
    POST /predict/json  — JSON-mode prediction (base64 image in body)
    GET  /health        — Server health and model status
    GET  /models/status — Detailed model availability check

Includes:
- File size validation
- MIME type validation
- Structured error responses
- Request ID tracking
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger

from backend.api.schemas import (
    BoundingBox,
    ErrorResponse,
    HealthResponse,
    ModelStatusResponse,
    PredictResponse,
)
from backend.config.settings import get_settings
from backend.utils.image_utils import base64_to_image, load_image

router = APIRouter()

# ── Constants ──
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/bmp",
    "image/tiff", "image/webp",
}


def _validate_upload(image: UploadFile, image_bytes: bytes) -> None:
    """Validate uploaded image file."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Size check
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(image_bytes)} bytes. "
                   f"Maximum: {max_bytes} bytes ({settings.max_upload_size_mb} MB).",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # MIME type check (best-effort)
    if image.content_type and image.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            f"Unexpected content type: {image.content_type}. "
            f"Proceeding anyway — Pillow will validate."
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server status and model loading state.",
)
async def health_check():
    """Check server health and model status."""
    settings = get_settings()

    try:
        from backend.models.loader import ModelLoader
        loader = ModelLoader()
        model_status = loader.status()
    except Exception:
        model_status = {"error": "Could not check model status"}

    return HealthResponse(
        status="ok",
        models_loaded=model_status,
        device=settings.resolved_device,
    )


@router.get(
    "/models/status",
    response_model=ModelStatusResponse,
    summary="Model availability",
    description="Check download and loading status of all models.",
)
async def model_status():
    """Get detailed model availability status."""
    try:
        from backend.models.model_manager import ModelManager
        manager = ModelManager()
        availability = manager.check_models()
        validation = manager.validate_models()
    except Exception as e:
        logger.warning(f"Model status check failed: {e}")
        availability = {"error": str(e)}
        validation = {}

    return ModelStatusResponse(
        availability=availability,
        validation=validation,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Medical VQA Prediction",
    description=(
        "Upload a medical image and ask a clinical question. "
        "Returns a text answer, explainability heatmap, and optional "
        "localization masks."
    ),
)
async def predict(
    question: str = Form(..., description="Medical question about the image"),
    image: UploadFile = File(..., description="Medical image file"),
    enable_gradcam: bool = Form(True),
    enable_attention: bool = Form(False),
    enable_localization: bool = Form(False),
    localization_prompt: Optional[str] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
):
    """
    Main prediction endpoint.

    Accepts a medical image (as file upload) and a question,
    returns a medical answer with optional explainability and
    localization outputs.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Prediction request: '{question[:80]}...'")

    try:
        # 1. Read and validate image
        image_bytes = await image.read()
        _validate_upload(image, image_bytes)

        pil_image = load_image(image_bytes)
        logger.info(f"[{request_id}] Image loaded: {pil_image.size}")

        # 2. Import orchestrator
        from backend.predict import run_prediction

        # 3. Run full pipeline
        result = run_prediction(
            image=pil_image,
            question=question,
            request_id=request_id,
            enable_gradcam=enable_gradcam,
            enable_attention=enable_attention,
            enable_localization=enable_localization,
            localization_prompt=localization_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return result

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"[{request_id}] Runtime error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}: {e}",
        )


@router.post(
    "/predict/json",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Medical VQA Prediction (JSON mode)",
    description="Send image as base64 in JSON body.",
)
async def predict_json(request: "PredictRequest"):
    """
    JSON-mode prediction endpoint.

    Accepts image as base64 string in the request body.
    Useful for programmatic access and testing.
    """
    # Lazy import to avoid circular dependency
    from backend.api.schemas import PredictRequest as PR

    request_id = str(uuid.uuid4())[:8]

    if not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_base64 is required for JSON mode.",
        )

    try:
        pil_image = base64_to_image(request.image_base64)

        from backend.predict import run_prediction

        result = run_prediction(
            image=pil_image,
            question=request.question,
            request_id=request_id,
            enable_gradcam=request.enable_gradcam,
            enable_attention=request.enable_attention,
            enable_localization=request.enable_localization,
            localization_prompt=request.localization_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error in JSON prediction")
        raise HTTPException(status_code=500, detail=str(e))


# Fix the type annotation for predict_json
from backend.api.schemas import PredictRequest
predict_json.__annotations__["request"] = PredictRequest
