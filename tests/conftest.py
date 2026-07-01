"""
Test fixtures for the Medical VQA System test suite.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.api.server import app
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a random RGB test image."""
    img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_data)


@pytest.fixture
def sample_image_bytes(sample_image):
    """Create test image as bytes (PNG)."""
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_question():
    """Create a sample medical question."""
    return "What abnormality is visible in this image?"


@pytest.fixture
def sample_heatmap():
    """Create a sample heatmap array."""
    return np.random.uniform(0, 1, (14, 14)).astype(np.float32)


@pytest.fixture
def sample_boxes():
    """Create sample bounding boxes."""
    return [
        {"x": 10, "y": 20, "w": 100, "h": 80, "score": 0.95, "label": "tumor"},
        {"x": 50, "y": 60, "w": 120, "h": 90, "score": 0.87, "label": "lesion"},
    ]


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    return mask
