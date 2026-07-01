"""
Pydantic schemas for API request/response validation.

Defines the data contracts for the Medical VQA API endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request schema for the /predict endpoint (JSON mode)."""

    image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded image string. Use this OR upload a file.",
    )
    question: str = Field(
        ...,
        description="Medical question about the image.",
        min_length=1,
        max_length=2000,
        examples=["What abnormality is visible?", "Where is the pneumonia?"],
    )
    enable_gradcam: bool = Field(
        True,
        description="Generate Grad-CAM heatmap.",
    )
    enable_attention: bool = Field(
        False,
        description="Generate attention rollout visualization.",
    )
    enable_localization: bool = Field(
        False,
        description="Enable Grounding DINO + SAM2 localization.",
    )
    localization_prompt: Optional[str] = Field(
        None,
        max_length=500,
        description=(
            "Text prompt for Grounding DINO (e.g., 'tumor'). "
            "If None, extracted from the question automatically."
        ),
    )
    max_new_tokens: Optional[int] = Field(
        None, ge=1, le=2048,
        description="Override max generation tokens.",
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Override sampling temperature.",
    )


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    """A detected bounding box."""
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str


class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    answer: str = Field(
        ..., description="Generated medical answer."
    )
    question: str = Field(
        ..., description="The input question (echoed back)."
    )
    inference_time_seconds: float = Field(
        ..., description="Time taken for VQA inference."
    )
    model_name: str = Field(
        default="stllava-med-7b",
        description="Name of the VQA model used.",
    )
    confidence: Optional[float] = Field(
        None, description="Confidence score (if available)."
    )

    # Explainability
    heatmap_url: Optional[str] = Field(
        None, description="URL to the Grad-CAM heatmap image."
    )
    overlay_url: Optional[str] = Field(
        None, description="URL to the heatmap overlay image."
    )
    attention_url: Optional[str] = Field(
        None, description="URL to the attention rollout image."
    )
    attention_overlay_url: Optional[str] = Field(
        None, description="URL to the attention overlay image."
    )

    # Localization
    bounding_boxes: Optional[list[BoundingBox]] = Field(
        None, description="Detected bounding boxes (from Grounding DINO)."
    )
    boxes_image_url: Optional[str] = Field(
        None, description="URL to annotated image with bounding boxes."
    )
    mask_overlay_url: Optional[str] = Field(
        None, description="URL to the segmentation mask overlay."
    )
    mask_urls: Optional[list[str]] = Field(
        None, description="URLs to individual segmentation masks."
    )

    # Metadata
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (device, params, etc.).",
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = "ok"
    models_loaded: dict = Field(
        default_factory=dict,
        description="Loading status of each model.",
    )
    device: str = Field(
        ..., description="Active compute device."
    )


class ModelStatusResponse(BaseModel):
    """Response schema for the /models/status endpoint."""

    availability: dict = Field(
        default_factory=dict,
        description="Download availability of each model component.",
    )
    validation: dict = Field(
        default_factory=dict,
        description="Validation results for downloaded models.",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
