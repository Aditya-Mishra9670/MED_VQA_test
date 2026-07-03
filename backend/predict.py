"""
Main prediction orchestrator for the Medical VQA pipeline.

Coordinates all components:
1. STLLaVA-Med → Medical Answer
2. Grad-CAM → Attention Heatmap
3. (Optional) Grounding DINO → SAM2 → Segmentation Mask

This is the central module called by both the API and CLI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger

from backend.api.schemas import BoundingBox, PredictResponse
from backend.config.settings import get_settings
from backend.models.inference import MedicalVQAInference
from backend.utils.logger import timed


# Singleton inference engine
_inference_engine: Optional[MedicalVQAInference] = None


def _get_engine() -> MedicalVQAInference:
    """Get or create the inference engine singleton."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = MedicalVQAInference(enable_cache=True)
    return _inference_engine


@timed
def run_prediction(
    image: Image.Image,
    question: str,
    request_id: str = "",
    enable_gradcam: bool = True,
    enable_attention: bool = False,
    enable_localization: bool = False,
    localization_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> PredictResponse:
    """
    Run the full Medical VQA prediction pipeline.

    Flow:
        Image + Question
            → STLLaVA-Med → Text Answer
            → Grad-CAM → Heatmap + Overlay
            → (Optional) Grounding DINO → Bounding Boxes
            → (Optional) SAM2 → Segmentation Masks

    Args:
        image: PIL Image of a medical scan.
        question: Clinical question about the image.
        request_id: Unique request identifier for logging.
        enable_gradcam: Whether to generate Grad-CAM heatmap.
        enable_attention: Whether to generate attention rollout.
        enable_localization: Whether to run Grounding DINO + SAM2.
        localization_prompt: Text prompt for Grounding DINO. Auto-extracted if None.
        max_new_tokens: Override max generation tokens.
        temperature: Override sampling temperature.

    Returns:
        PredictResponse with all generated outputs.
    """
    settings = get_settings()
    engine = _get_engine()
    prefix = f"{request_id}_" if request_id else ""

    logger.info(f"[{request_id}] Starting prediction pipeline")
    pipeline_start = time.perf_counter()

    # Ensure image is RGB
    image = image.convert("RGB")

    # ─── Step 1: Medical VQA (STLLaVA-Med) ───
    logger.info(f"[{request_id}] Step 1: Running STLLaVA-Med inference...")
    vqa_result = engine.predict(
        image=image,
        question=question,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    answer_preview = vqa_result.answer[:100] if vqa_result.answer else "(empty)"
    logger.info(
        f"[{request_id}] Answer: {answer_preview}... "
        f"({vqa_result.inference_time_seconds:.2f}s)"
    )

    # Initialize response fields
    heatmap_url = None
    overlay_url = None
    attention_url = None
    attention_overlay_url = None
    bounding_boxes = None
    boxes_image_url = None
    mask_overlay_url = None
    mask_urls = None

    # ─── Step 2: Grad-CAM Explainability ───
    if enable_gradcam:
        try:
            logger.info(f"[{request_id}] Step 2: Running Grad-CAM...")
            from backend.explainability.gradcam import GradCAMExplainer
            from backend.models.loader import ModelLoader

            loader = ModelLoader()
            vision_encoder = loader.stllava.get_vision_encoder()

            if vision_encoder is not None:
                image_tensor = loader.stllava.preprocess_image(image)

                explainer = GradCAMExplainer(
                    model=vision_encoder,
                    use_vit=True,
                    reshape_height=24,
                    reshape_width=24,
                )

                gradcam_result = explainer.explain(
                    image=image,
                    input_tensor=image_tensor,
                    output_dir=Path(settings.heatmaps_dir),
                    prefix=prefix,
                )

                heatmap_url = f"/outputs/heatmaps/{prefix}heatmap.png"
                overlay_url = f"/outputs/heatmaps/{prefix}overlay.png"
            else:
                logger.warning(f"[{request_id}] Vision encoder not available for Grad-CAM")

        except Exception as e:
            logger.warning(f"[{request_id}] Grad-CAM failed: {e}")

    # ─── Step 3: Attention Rollout ───
    if enable_attention:
        try:
            logger.info(f"[{request_id}] Step 3: Running Attention Rollout...")
            from backend.explainability.attention import AttentionRollout
            from backend.models.loader import ModelLoader

            loader = ModelLoader()
            vision_encoder = loader.stllava.get_vision_encoder()

            if vision_encoder is not None:
                image_tensor = loader.stllava.preprocess_image(image)

                rollout = AttentionRollout(model=vision_encoder)
                attn_result = rollout.explain(
                    image=image,
                    input_tensor=image_tensor,
                    output_dir=Path(settings.heatmaps_dir),
                    prefix=prefix,
                )

                attention_url = f"/outputs/heatmaps/{prefix}attention.png"
                attention_overlay_url = f"/outputs/heatmaps/{prefix}attention_overlay.png"
            else:
                logger.warning(f"[{request_id}] Vision encoder not available for attention rollout")

        except Exception as e:
            logger.warning(f"[{request_id}] Attention rollout failed: {e}")

    # ─── Step 4: Localization (Grounding DINO + SAM2) ───
    if enable_localization:
        try:
            logger.info(f"[{request_id}] Step 4: Running localization...")
            from backend.models.loader import ModelLoader

            loader = ModelLoader()

            # Determine text prompt for Grounding DINO
            text_prompt = localization_prompt
            if text_prompt is None:
                text_prompt = _extract_localization_prompt(question, vqa_result.answer)

            # Grounding DINO: text → bounding boxes
            logger.info(f"[{request_id}] Grounding DINO prompt: '{text_prompt}'")
            gdino = loader.grounding_dino
            gdino_result = gdino.localize(
                image=image,
                text_prompt=text_prompt,
                output_dir=Path(settings.masks_dir),
                prefix=prefix,
            )

            bounding_boxes = [
                BoundingBox(**box) for box in gdino_result["boxes"]
            ]
            boxes_image_url = f"/outputs/masks/{prefix}boxes.png"

            # SAM2: bounding boxes → segmentation masks
            if gdino_result["boxes"]:
                logger.info(
                    f"[{request_id}] SAM2: Segmenting "
                    f"{len(gdino_result['boxes'])} regions..."
                )
                sam2 = loader.sam2
                sam2_result = sam2.segment_and_visualize(
                    image=image,
                    boxes=gdino_result["boxes"],
                    output_dir=Path(settings.masks_dir),
                    prefix=prefix,
                )

                mask_overlay_url = f"/outputs/masks/{prefix}mask_overlay.png"
                mask_urls = [
                    f"/outputs/masks/{Path(p).name}"
                    for p in sam2_result.get("mask_paths", [])
                ]

        except Exception as e:
            logger.warning(f"[{request_id}] Localization failed: {e}")

    # ─── Build Response ───
    total_time = time.perf_counter() - pipeline_start
    logger.info(f"[{request_id}] Pipeline complete in {total_time:.2f}s")

    return PredictResponse(
        answer=vqa_result.answer,
        question=question,
        inference_time_seconds=vqa_result.inference_time_seconds,
        model_name=vqa_result.model_name,
        confidence=vqa_result.confidence,
        heatmap_url=heatmap_url,
        overlay_url=overlay_url,
        attention_url=attention_url,
        attention_overlay_url=attention_overlay_url,
        bounding_boxes=bounding_boxes,
        boxes_image_url=boxes_image_url,
        mask_overlay_url=mask_overlay_url,
        mask_urls=mask_urls,
        metadata={
            "request_id": request_id,
            "total_pipeline_seconds": round(total_time, 3),
            "device": settings.resolved_device,
            "gradcam_enabled": enable_gradcam,
            "attention_enabled": enable_attention,
            "localization_enabled": enable_localization,
        },
    )


def _extract_localization_prompt(question: str, answer: str) -> str:
    """
    Extract a localization prompt from the question and answer.

    Simple heuristic: look for medical terms in the answer.
    Falls back to common terms from the question.

    Args:
        question: The original medical question.
        answer: The generated medical answer.

    Returns:
        Text prompt suitable for Grounding DINO.
    """
    # Common medical terms to look for
    medical_terms = [
        "tumor", "lesion", "mass", "nodule", "opacity",
        "fracture", "pneumonia", "effusion", "cardiomegaly",
        "consolidation", "infiltrate", "edema", "hemorrhage",
        "calcification", "stenosis", "aneurysm", "polyp",
        "cyst", "abscess", "fibrosis", "atelectasis",
    ]

    # Check answer first, then question
    text = f"{answer} {question}".lower()
    found = [term for term in medical_terms if term in text]

    if found:
        return ". ".join(found)

    # Fallback: use "abnormality" as a generic prompt
    return "abnormality"
