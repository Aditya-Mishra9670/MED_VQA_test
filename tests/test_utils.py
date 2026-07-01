"""
Tests for utility modules: image processing, device detection,
output management, and startup validation.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestImageUtils:
    """Tests for image processing utilities."""

    def test_load_image_from_bytes(self, sample_image_bytes):
        from backend.utils.image_utils import load_image
        img = load_image(sample_image_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_image_rejects_empty_bytes(self):
        from backend.utils.image_utils import load_image
        with pytest.raises(ValueError, match="Empty image data"):
            load_image(b"")

    def test_load_image_rejects_invalid_data(self):
        from backend.utils.image_utils import load_image
        with pytest.raises(ValueError, match="Could not open image"):
            load_image(b"not an image")

    def test_load_image_from_path(self, sample_image, tmp_path):
        from backend.utils.image_utils import load_image
        path = tmp_path / "test.png"
        sample_image.save(path)
        img = load_image(str(path))
        assert isinstance(img, Image.Image)

    def test_image_to_base64(self, sample_image):
        from backend.utils.image_utils import image_to_base64
        b64 = image_to_base64(sample_image)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_base64_roundtrip(self, sample_image):
        from backend.utils.image_utils import image_to_base64, base64_to_image
        b64 = image_to_base64(sample_image)
        decoded = base64_to_image(b64)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == sample_image.size

    def test_base64_with_data_uri(self, sample_image):
        from backend.utils.image_utils import image_to_base64, base64_to_image
        b64 = image_to_base64(sample_image)
        data_uri = f"data:image/png;base64,{b64}"
        decoded = base64_to_image(data_uri)
        assert isinstance(decoded, Image.Image)

    def test_create_heatmap_overlay(self, sample_image, sample_heatmap):
        from backend.utils.image_utils import create_heatmap_overlay
        overlay = create_heatmap_overlay(sample_image, sample_heatmap)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == sample_image.size

    def test_create_mask_overlay(self, sample_image, sample_mask):
        from backend.utils.image_utils import create_mask_overlay
        overlay = create_mask_overlay(sample_image, sample_mask)
        assert isinstance(overlay, Image.Image)

    def test_draw_bounding_boxes(self, sample_image, sample_boxes):
        from backend.utils.image_utils import draw_bounding_boxes
        annotated = draw_bounding_boxes(sample_image, sample_boxes)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == sample_image.size

    def test_save_image(self, sample_image, tmp_path):
        from backend.utils.image_utils import save_image
        path = save_image(sample_image, tmp_path / "sub" / "test.png")
        assert path.exists()
        assert path.stat().st_size > 0


class TestDeviceUtils:
    """Tests for device detection utilities."""

    def test_get_device_cpu(self):
        from backend.utils.device import get_device
        device = get_device("cpu")
        assert str(device) == "cpu"

    def test_get_device_auto(self):
        from backend.utils.device import get_device
        device = get_device("auto")
        assert str(device) in ["cpu", "cuda", "mps"]

    def test_get_device_info(self):
        from backend.utils.device import get_device_info
        info = get_device_info()
        assert isinstance(info, dict)
        assert "pytorch_version" in info
        assert "cuda_available" in info

    def test_clear_gpu_memory(self):
        from backend.utils.device import clear_gpu_memory
        # Should not raise even without GPU
        clear_gpu_memory()


class TestOutputManager:
    """Tests for output management utilities."""

    def test_get_output_path(self, tmp_path):
        from backend.utils.output_manager import OutputManager
        manager = OutputManager(tmp_path)
        path = manager.get_output_path("heatmaps", "test.png")
        assert str(path).startswith(str(tmp_path))

    def test_path_traversal_protection(self, tmp_path):
        from backend.utils.output_manager import OutputManager
        manager = OutputManager(tmp_path)
        with pytest.raises(ValueError):
            manager.get_output_path("../../../etc", "passwd")

    def test_generate_unique_prefix(self, tmp_path):
        from backend.utils.output_manager import OutputManager
        manager = OutputManager(tmp_path)
        p1 = manager.generate_unique_prefix()
        p2 = manager.generate_unique_prefix()
        assert p1 != p2
        assert p1.endswith("_")

    def test_disk_usage(self, tmp_path):
        from backend.utils.output_manager import OutputManager
        manager = OutputManager(tmp_path)
        usage = manager.get_disk_usage()
        assert "total_size_mb" in usage
        assert "file_count" in usage

    def test_cleanup_empty_dir(self, tmp_path):
        from backend.utils.output_manager import OutputManager
        manager = OutputManager(tmp_path)
        deleted = manager.cleanup_old_outputs()
        assert deleted == 0


class TestSettings:
    """Tests for configuration settings."""

    def test_get_settings(self):
        from backend.config.settings import get_settings
        settings = get_settings()
        assert settings is not None
        assert settings.api_port == 8000

    def test_resolved_device(self):
        from backend.config.settings import get_settings
        settings = get_settings()
        device = settings.resolved_device
        assert device in ["cuda", "mps", "cpu"]

    def test_output_directories(self):
        from backend.config.settings import get_settings
        settings = get_settings()
        assert settings.answers_dir.exists()
        assert settings.heatmaps_dir.exists()
        assert settings.masks_dir.exists()


class TestStartupChecks:
    """Tests for startup validation."""

    def test_run_startup_checks(self):
        from backend.utils.startup import run_startup_checks
        results = run_startup_checks()
        assert isinstance(results, dict)
        assert "python" in results
        assert "dependencies" in results
        assert results["python"]["ok"] is True
