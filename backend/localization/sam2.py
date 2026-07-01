"""
SAM2 (Segment Anything Model 2) wrapper for precise segmentation.

Converts coarse bounding boxes from Grounding DINO into pixel-level
segmentation masks. Provides exact lesion/region boundaries.

Includes automatic checkpoint download and graceful error handling.

Reference: https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from backend.utils.image_utils import create_mask_overlay, save_image
from backend.utils.logger import timed


class SAM2Wrapper:
    """
    SAM2 model wrapper for bounding-box-prompted segmentation.

    Takes bounding boxes (from Grounding DINO) and produces
    pixel-accurate segmentation masks.

    Outputs:
    - Binary segmentation mask
    - Mask overlay on original image
    - Contour visualization
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
        config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
    ):
        """
        Initialize SAM2 wrapper.

        Args:
            checkpoint_path: Path to SAM2 model checkpoint.
            config_path: Path to SAM2 config YAML.
            device: Device for inference.
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.predictor = None
        self._loaded = False

    def load(self) -> None:
        """Load SAM2 model."""
        if self._loaded:
            logger.info("SAM2 already loaded.")
            return

        # Ensure checkpoint exists
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            logger.info("SAM2 checkpoint not found, attempting download...")
            try:
                from backend.models.model_manager import ModelManager
                manager = ModelManager()
                result = manager.ensure_sam2_available()
                if result:
                    self.checkpoint_path = result
                    checkpoint = Path(result)
            except Exception as e:
                raise RuntimeError(
                    f"SAM2 checkpoint not found at {self.checkpoint_path} "
                    f"and auto-download failed: {e}"
                )

        if not checkpoint.exists():
            raise RuntimeError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}. "
                f"Run: python -m backend.models.model_manager to download."
            )

        logger.info(f"Loading SAM2 from {self.checkpoint_path}...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                self.config_path,
                self.checkpoint_path,
                device=self.device,
            )
            self.predictor = SAM2ImagePredictor(sam2_model)
            self._loaded = True
            logger.info("SAM2 loaded successfully.")

        except ImportError as e:
            logger.error(
                f"SAM2 package not installed. "
                f"Install from: https://github.com/facebookresearch/sam2 — {e}"
            )
            raise RuntimeError(
                "SAM2 package not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e

    @timed
    def segment_box(
        self,
        image: Image.Image,
        box: dict,
    ) -> np.ndarray:
        """
        Segment a region defined by a bounding box.

        Args:
            image: PIL Image to segment.
            box: Bounding box dict with keys 'x', 'y', 'w', 'h'.

        Returns:
            Binary mask as numpy array (H, W) with values 0 or 1.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        img_np = np.array(image.convert("RGB"))
        self.predictor.set_image(img_np)

        # Convert box format: (x, y, w, h) → (x1, y1, x2, y2)
        input_box = np.array([
            box["x"],
            box["y"],
            box["x"] + box["w"],
            box["y"] + box["h"],
        ])

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Return the best mask
        return masks[0].astype(np.uint8)

    @timed
    def segment_boxes(
        self,
        image: Image.Image,
        boxes: list[dict],
    ) -> list[np.ndarray]:
        """
        Segment multiple regions from bounding boxes.

        Args:
            image: PIL Image to segment.
            boxes: List of bounding box dicts.

        Returns:
            List of binary masks (one per box).
        """
        masks = []
        for box in boxes:
            try:
                mask = self.segment_box(image, box)
                masks.append(mask)
            except Exception as e:
                logger.warning(f"Failed to segment box {box}: {e}")
                # Create empty mask as fallback
                img_np = np.array(image)
                masks.append(np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
        return masks

    @timed
    def segment_and_visualize(
        self,
        image: Image.Image,
        boxes: list[dict],
        output_dir: Path,
        prefix: str = "",
        colors: Optional[list[tuple[int, int, int]]] = None,
    ) -> dict:
        """
        Full segmentation pipeline: segment all boxes and save visualizations.

        Args:
            image: Input PIL Image.
            boxes: List of bounding box dicts from Grounding DINO.
            output_dir: Directory to save output files.
            prefix: Filename prefix.
            colors: Optional list of RGB colors for each mask.

        Returns:
            Dict with mask paths, overlay path, and raw masks.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not boxes:
            logger.warning("No bounding boxes provided for segmentation.")
            return {"masks": [], "mask_paths": [], "overlay_path": None, "num_masks": 0}

        # Default color palette
        default_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        colors = colors or default_colors

        masks = self.segment_boxes(image, boxes)
        mask_paths = []

        # Save individual masks
        for i, mask in enumerate(masks):
            mask_img = Image.fromarray(mask * 255)
            path = save_image(
                mask_img, output_dir / f"{prefix}mask_{i}.png"
            )
            mask_paths.append(str(path))

        # Create combined overlay
        overlay = image.copy().convert("RGB")
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            overlay = create_mask_overlay(overlay, mask, color=color, alpha=0.4)

        overlay_path = save_image(
            overlay, output_dir / f"{prefix}mask_overlay.png"
        )

        # Create combined binary mask
        combined_path = None
        if masks:
            combined = np.zeros_like(masks[0])
            for mask in masks:
                combined = np.maximum(combined, mask)
            combined_img = Image.fromarray(combined * 255)
            combined_path = save_image(
                combined_img, output_dir / f"{prefix}mask_combined.png"
            )

        logger.info(
            f"SAM2 segmented {len(masks)} regions. "
            f"Overlay saved to {overlay_path}"
        )

        return {
            "mask_paths": mask_paths,
            "overlay_path": str(overlay_path),
            "combined_mask_path": str(combined_path) if combined_path else None,
            "masks": masks,
            "num_masks": len(masks),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
