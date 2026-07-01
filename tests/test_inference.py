"""
Tests for inference, model management, and VQA result serialization.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.models.inference import VQAResult


class TestVQAResult:
    """Tests for the VQAResult dataclass."""

    def test_to_dict(self):
        result = VQAResult(
            answer="Pneumonia in right lung.",
            question="What is wrong?",
            inference_time_seconds=1.234,
        )
        d = result.to_dict()
        assert d["answer"] == "Pneumonia in right lung."
        assert d["question"] == "What is wrong?"
        assert d["inference_time_seconds"] == 1.234
        assert d["model_name"] == "stllava-med-7b"

    def test_default_values(self):
        result = VQAResult(
            answer="test",
            question="test?",
            inference_time_seconds=0.5,
        )
        assert result.confidence is None
        assert result.metadata == {}
        assert result.model_name == "stllava-med-7b"

    def test_custom_metadata(self):
        result = VQAResult(
            answer="test",
            question="test?",
            inference_time_seconds=0.5,
            metadata={"device": "cuda", "tokens": 100},
        )
        d = result.to_dict()
        assert d["metadata"]["device"] == "cuda"
        assert d["metadata"]["tokens"] == 100

    def test_confidence_score(self):
        result = VQAResult(
            answer="test",
            question="test?",
            inference_time_seconds=0.5,
            confidence=0.95,
        )
        assert result.confidence == 0.95


class TestModelManager:
    """Tests for the ModelManager class."""

    def test_check_models_returns_dict(self):
        from backend.models.model_manager import ModelManager
        manager = ModelManager()
        status = manager.check_models()
        assert isinstance(status, dict)
        assert "stllava" in status
        assert "llava_package" in status

    def test_validate_models_returns_dict(self):
        from backend.models.model_manager import ModelManager
        manager = ModelManager()
        results = manager.validate_models()
        assert isinstance(results, dict)
        assert "stllava" in results


class TestModelLoader:
    """Tests for the ModelLoader class."""

    def test_singleton_pattern(self):
        from backend.models.loader import ModelLoader
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        assert loader1 is loader2

    def test_status_returns_dict(self):
        from backend.models.loader import ModelLoader
        loader = ModelLoader()
        status = loader.status()
        assert isinstance(status, dict)
        assert "stllava_loaded" in status
        assert "device" in status

    def test_unload_resets_state(self):
        from backend.models.loader import ModelLoader
        loader = ModelLoader()
        loader.unload_all()
        # After unload, getting a new loader should work
        loader2 = ModelLoader()
        status = loader2.status()
        assert status["stllava_loaded"] is False
