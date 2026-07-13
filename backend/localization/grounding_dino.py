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

        pass # Replaced with HuggingFace implementation

        logger.info(f"Loading Grounding DINO natively via HuggingFace transformers (bypassing compilation)...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        # Load the HuggingFace version of groundingdino_swint_ogc (IDEA-Research/grounding-dino-tiny)
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        self._loaded = True
        logger.info("Grounding DINO loaded successfully via transformers.")

    @timed
    def predict(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        # Ensure text prompt is properly formatted with trailing period for HF version
        if not text_prompt.endswith("."):
            text_prompt += "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results_hf = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[image.size[::-1]]
        )[0]
        
        results = []
        for score, label, box in zip(results_hf["scores"], results_hf["labels"], results_hf["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            
            results.append({
                "x": int(max(0, x1)),
                "y": int(max(0, y1)),
                "w": int(max(0, x2 - x1)),
                "h": int(max(0, y2 - y1)),
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
