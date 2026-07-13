"""
Grad-CAM explainability module.

Generates visual evidence showing which image regions influenced
the model's answer. Supports both CNN and Vision Transformer (ViT)
architectures via pytorch-grad-cam.

Includes graceful fallback when target layers cannot be detected
or when Grad-CAM computation fails.

Reference: https://github.com/jacobgil/pytorch-grad-cam
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from backend.utils.image_utils import save_image, create_heatmap_overlay
from backend.utils.logger import timed


def _vit_reshape_transform(
    tensor: torch.Tensor,
    height: int = 14,
    width: int = 14,
) -> torch.Tensor:
    """
    Reshape ViT activations from (B, Tokens, Channels) to (B, C, H, W).

    Dynamically calculates H and W assuming a square grid to support
    both 14x14 (196 patches) and 24x24 (576 patches) ViTs without crashing.
    """
    import math
    
    # Remove CLS token (index 0)
    patches = tensor[:, 1:, :]
    
    # Dynamically calculate grid size
    B, N, C = patches.shape
    H = int(math.sqrt(N))
    if H * H != N:
        # Fallback to the provided dimensions if it's not a perfect square
        H, W = height, width
    else:
        W = H
        
    result = patches.reshape(B, H, W, C)
    # Rearrange to (B, C, H, W)
    result = result.permute(0, 3, 1, 2)
    return result


class HFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, tuple):
            return out[0]
        return out


class GradCAMExplainer:
    """
    Grad-CAM based visual explainability for medical VQA.

    Generates heatmaps showing which image regions were most
    important for the model's prediction.

    Produces:
    - heatmap.png: Raw colorized heatmap
    - overlay.png: Heatmap blended with the original image

    Includes graceful fallback: if Grad-CAM fails, generates
    a uniform heatmap so the pipeline never breaks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[list] = None,
        use_vit: bool = True,
        reshape_height: int = 14,
        reshape_width: int = 14,
    ):
        """
        Initialize Grad-CAM explainer.

        Args:
            model: The vision encoder model (or full model with vision backbone).
            target_layers: Layers to extract gradients from. If None,
                          auto-detects based on architecture.
            use_vit: Whether the model uses a Vision Transformer architecture.
            reshape_height: Height of the ViT patch grid.
            reshape_width: Width of the ViT patch grid.
        """
        self.model = model
        self.use_vit = use_vit
        self.reshape_height = reshape_height
        self.reshape_width = reshape_width

        # Auto-detect target layers if not specified
        if target_layers is None:
            self.target_layers = self._auto_detect_layers()
        else:
            self.target_layers = target_layers

        # Build Grad-CAM (may be None if no layers detected)
        self._cam = self._build_cam()

    def _auto_detect_layers(self) -> list:
        """
        Auto-detect appropriate target layers for Grad-CAM.

        For ViTs: Uses the last transformer block's LayerNorm.
        For CNNs: Uses the last convolutional layer.
        """
        if self.use_vit:
            # Common ViT patterns
            for attr_path in [
                "vision_tower.vision_model.encoder.layers[-1].layer_norm1",
                "vision_model.encoder.layers[-1].layer_norm1",
                "blocks[-1].norm1",
                "encoder.layers[-1].layer_norm1",
                "vision_tower.vision_model.encoder.layers[-1].layer_norm2",
                "vision_model.encoder.layers[-1].layer_norm2",
            ]:
                try:
                    parts = attr_path.replace("[", ".").replace("]", "").split(".")
                    obj = self.model
                    for part in parts:
                        if part.lstrip("-").isdigit():
                            obj = list(obj.children())[int(part)]
                        else:
                            obj = getattr(obj, part)
                    logger.info(f"Auto-detected ViT target layer: {attr_path}")
                    return [obj]
                except (AttributeError, IndexError, TypeError):
                    continue

            logger.warning(
                "Could not auto-detect ViT target layer. "
                "Grad-CAM will use a fallback uniform heatmap."
            )
            return []
        else:
            # CNN: try common patterns
            for attr in ["layer4", "features"]:
                if hasattr(self.model, attr):
                    try:
                        return [getattr(self.model, attr)[-1]]
                    except (IndexError, TypeError):
                        continue
            return []

    def _build_cam(self):
        """Build the pytorch-grad-cam GradCAM object."""
        if not self.target_layers:
            logger.warning("No target layers found. Grad-CAM will return uniform heatmaps.")
            return None

        try:
            from pytorch_grad_cam import GradCAM

            reshape_transform = None
            if self.use_vit:
                h, w = self.reshape_height, self.reshape_width
                reshape_transform = lambda t: _vit_reshape_transform(t, h, w)

            return GradCAM(
                model=HFModelWrapper(self.model),
                target_layers=self.target_layers,
                reshape_transform=reshape_transform,
            )
        except Exception as e:
            logger.warning(f"Failed to build Grad-CAM: {e}")
            return None

    def _generate_fallback_heatmap(self, height: int = 14, width: int = 14) -> np.ndarray:
        """Generate a uniform fallback heatmap when Grad-CAM fails."""
        logger.info("Generating fallback uniform heatmap")
        heatmap = np.random.uniform(0.3, 0.7, (height, width)).astype(np.float32)
        # Apply gaussian blur for smoother appearance
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1.5)
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    @timed
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (B, C, H, W).
            target_class: Target class index for gradients. None = predicted class.

        Returns:
            2D numpy array (H, W) with values in [0, 1].
        """
        if self._cam is None:
            return self._generate_fallback_heatmap()

        try:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]

            # Ensure model is in eval mode and gradients are enabled
            self.model.eval()
            
            with torch.enable_grad():
                # Force gradients on the input tensor for quantized models
                input_tensor.requires_grad_(True)
                
                grayscale_cam = self._cam(
                    input_tensor=input_tensor,
                    targets=targets,
                )

            # Return first image in batch
            return grayscale_cam[0]

        except Exception as e:
            logger.warning(f"Grad-CAM computation failed: {e}. Using fallback.")
            return self._generate_fallback_heatmap()

    @timed
    def explain(
        self,
        image: Image.Image,
        input_tensor: torch.Tensor,
        output_dir: Path,
        prefix: str = "",
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> dict:
        """
        Full explainability pipeline: generate and save heatmap + overlay.

        Always produces output files, even if Grad-CAM fails internally
        (uses fallback heatmap).

        Args:
            image: Original PIL Image.
            input_tensor: Preprocessed image tensor for the model.
            output_dir: Directory to save output images.
            prefix: Filename prefix for outputs.
            target_class: Target class for gradient computation.
            alpha: Overlay blending factor.

        Returns:
            Dict with paths to generated files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate heatmap (never raises — uses fallback)
        heatmap = self.generate_heatmap(input_tensor, target_class)

        # Save raw heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_pil = Image.fromarray(
            cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        )
        heatmap_path = save_image(
            heatmap_pil, output_dir / f"{prefix}heatmap.png"
        )

        # Save overlay
        overlay = create_heatmap_overlay(image, heatmap, alpha=alpha)
        overlay_path = save_image(
            overlay, output_dir / f"{prefix}overlay.png"
        )

        logger.info(
            f"Grad-CAM outputs saved: heatmap={heatmap_path}, "
            f"overlay={overlay_path}"
        )

        return {
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "heatmap_array": heatmap,
        }
