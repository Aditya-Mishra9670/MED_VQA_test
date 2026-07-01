"""
Grounding DINO wrapper for text-guided object localization.

Converts natural language descriptions (e.g., "tumor", "fracture")
into bounding box coordinates on medical images.

Includes automatic checkpoint download and graceful fallback
when the GroundingDINO package is not installed.

Reference: https://github.com/IDEA-Research/GroundingDINO
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from backend.utils.image_utils import draw_bounding_boxes, save_image
from backend.utils.logger import timed


class GroundingDINOWrapper:
    """
    Grounding DINO model wrapper for text-to-region localization.

    Takes a text prompt (e.g., "tumor") and an image, returns
    bounding boxes around matching regions.

    Outputs:
    - List of bounding boxes with confidence scores
    - Annotated image with drawn boxes
    """

    def __init__(
        self,
        config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "checkpoints/groundingdino_swint_ogc.pth",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Grounding DINO wrapper.

        Args:
            config_path: Path to Grounding DINO config file.
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            box_threshold: Confidence threshold for bounding boxes.
            text_threshold: Confidence threshold for text grounding.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load Grounding DINO model from checkpoint."""
        if self._loaded:
            logger.info("Grounding DINO already loaded.")
            return

        # Ensure checkpoint exists
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            logger.info("Grounding DINO checkpoint not found, attempting download...")
            try:
                from backend.models.model_manager import ModelManager
                manager = ModelManager()
                result = manager.ensure_grounding_dino_available()
                if result:
                    self.checkpoint_path = result
                    checkpoint = Path(result)
            except Exception as e:
                raise RuntimeError(
                    f"Grounding DINO checkpoint not found at {self.checkpoint_path} "
                    f"and auto-download failed: {e}"
                )

        if not checkpoint.exists():
            raise RuntimeError(
                f"Grounding DINO checkpoint not found: {self.checkpoint_path}. "
                f"Run: python -m backend.models.model_manager to download."
            )

        logger.info(f"Loading Grounding DINO from {self.checkpoint_path}...")

        try:
            from groundingdino.util.inference import load_model

            self.model = load_model(
                self.config_path,
                self.checkpoint_path,
                device=self.device,
            )
            self._loaded = True
            logger.info("Grounding DINO loaded successfully.")

        except ImportError as e:
            logger.error(
                f"Grounding DINO package not installed. "
                f"Install from: https://github.com/IDEA-Research/GroundingDINO — {e}"
            )
            raise RuntimeError(
                "GroundingDINO package not installed. "
                "Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            ) from e

    @timed
    def predict(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Detect regions matching the text prompt.

        Args:
            image: PIL Image to search in.
            text_prompt: Text description of the target (e.g., "tumor").
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text confidence threshold.

        Returns:
            List of detected regions, each as:
            {"x": int, "y": int, "w": int, "h": int, "score": float, "label": str}
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        from groundingdino.util.inference import (
            load_image as gdino_load_image,
            predict as gdino_predict,
        )

        # Convert PIL to format expected by Grounding DINO
        # Use a safe temp file with proper cleanup
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix="gdino_"
            ) as f:
                tmp_path = f.name
                image.save(tmp_path)

            image_source, image_transformed = gdino_load_image(tmp_path)
        finally:
            # Always clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Run prediction
        boxes, logits, phrases = gdino_predict(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            device=self.device,
        )

        # Convert normalized boxes to pixel coordinates
        h, w = image_source.shape[:2]
        results = []

        for box, score, label in zip(boxes, logits, phrases):
            # box format: cx, cy, w, h (normalized)
            cx, cy, bw, bh = box.tolist()
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            box_w = int(bw * w)
            box_h = int(bh * h)

            results.append({
                "x": max(0, x),
                "y": max(0, y),
                "w": min(box_w, w - max(0, x)),
                "h": min(box_h, h - max(0, y)),
                "score": round(float(score), 4),
                "label": label,
            })

        logger.info(
            f"Grounding DINO found {len(results)} regions for '{text_prompt}'"
        )
        return results

    @timed
    def localize(
        self,
        image: Image.Image,
        text_prompt: str,
        output_dir: Path,
        prefix: str = "",
    ) -> dict:
        """
        Full localization pipeline: detect and visualize.

        Args:
            image: Input image.
            text_prompt: Text query for localization.
            output_dir: Directory to save annotated image.
            prefix: Filename prefix.

        Returns:
            Dict with bounding boxes and annotated image path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        boxes = self.predict(image, text_prompt)

        # Draw boxes on image
        annotated = draw_bounding_boxes(image, boxes)
        annotated_path = save_image(
            annotated, output_dir / f"{prefix}boxes.png"
        )

        return {
            "boxes": boxes,
            "annotated_path": str(annotated_path),
            "num_detections": len(boxes),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
