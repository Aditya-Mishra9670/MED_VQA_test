"""
Image processing utilities.

Handles loading, preprocessing, encoding, and visualization overlays
used across the VQA pipeline. Includes input sanitization and
safe file handling.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# Maximum image dimensions to prevent memory exhaustion
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels


def load_image(source: Union[str, Path, bytes]) -> Image.Image:
    """
    Load an image from a file path, URL, or raw bytes.

    Includes validation for image dimensions and pixel count.

    Args:
        source: File path string/Path, or raw bytes (e.g., from upload).

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file path doesn't exist.
        ValueError: If the image is too large or corrupt.
    """
    # Set PIL decompression bomb limit
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    try:
        if isinstance(source, bytes):
            if len(source) == 0:
                raise ValueError("Empty image data")
            img = Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {source}")
            img = Image.open(path).convert("RGB")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
    except Image.DecompressionBombError:
        raise ValueError(
            f"Image exceeds maximum pixel count ({MAX_IMAGE_PIXELS:,} pixels). "
            f"Please resize the image."
        )
    except (Image.UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Could not open image: {e}")

    # Validate dimensions
    w, h = img.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        logger.warning(
            f"Image too large ({w}x{h}), resizing to {MAX_IMAGE_DIMENSION}px max"
        )
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

    return img


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image: PIL Image to encode.
        fmt: Image format (PNG, JPEG, etc.).

    Returns:
        Base64-encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image.

    Args:
        b64_string: Base64-encoded image string.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the string is not valid base64 or not an image.
    """
    try:
        # Handle data URI prefix if present
        if "," in b64_string and b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_string)
        return load_image(image_bytes)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Overlay a heatmap on an image using OpenCV colormap.

    Args:
        image: Original PIL Image.
        heatmap: 2D numpy array (float, 0-1 range) — the activation map.
        alpha: Blending weight for the overlay (0=image only, 1=heatmap only).
        colormap: OpenCV colormap constant.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Convert PIL to numpy BGR
    img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_np.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))

    # Normalize to 0-255
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)

    # Convert back to PIL RGB
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def create_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay a binary segmentation mask on an image.

    Args:
        image: Original PIL Image.
        mask: 2D binary numpy array (0 or 1).
        color: RGB color tuple for the mask overlay.
        alpha: Blending weight for the mask region.

    Returns:
        PIL Image with mask overlay.
    """
    img_np = np.array(image.convert("RGB")).copy()
    h, w = img_np.shape[:2]

    # Resize mask to match image
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))

    # Create colored mask
    colored_mask = np.zeros_like(img_np)
    colored_mask[mask_resized > 0] = color

    # Blend only in mask region
    mask_region = mask_resized > 0
    if mask_region.any():
        img_np[mask_region] = cv2.addWeighted(
            img_np[mask_region], 1 - alpha,
            colored_mask[mask_region], alpha,
            0,
        )

    return Image.fromarray(img_np)


def draw_bounding_boxes(
    image: Image.Image,
    boxes: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: PIL Image to draw on.
        boxes: List of dicts with keys 'x', 'y', 'w', 'h' and optional 'label', 'score'.
        color: BGR color tuple for the box.
        thickness: Line thickness in pixels.

    Returns:
        PIL Image with bounding boxes drawn.
    """
    img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    for box in boxes:
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        cv2.rectangle(img_np, (x, y), (x + w, y + h), color, thickness)

        # Add label if present
        label_parts = []
        if "label" in box:
            label_parts.append(box["label"])
        if "score" in box:
            label_parts.append(f"{box['score']:.2f}")

        if label_parts:
            label = " | ".join(label_parts)
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(
                img_np, (x, y - text_h - 6), (x + text_w + 4, y), color, -1
            )
            cv2.putText(
                img_np, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness,
            )

    return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))


def save_image(image: Image.Image, path: Union[str, Path]) -> Path:
    """
    Save a PIL Image to disk and return the path.

    Creates parent directories if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
