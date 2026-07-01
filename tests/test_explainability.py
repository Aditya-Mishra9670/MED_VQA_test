"""
Tests for explainability modules: Grad-CAM and Attention Rollout.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


class SimpleCNN(nn.Module):
    """Simple CNN for testing Grad-CAM."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class TestGradCAMExplainer:
    """Tests for the GradCAM explainability module."""

    def test_gradcam_with_cnn(self, tmp_path):
        from backend.explainability.gradcam import GradCAMExplainer

        model = SimpleCNN()
        explainer = GradCAMExplainer(
            model=model,
            target_layers=[model.features[-2]],  # Conv layer
            use_vit=False,
        )

        input_tensor = torch.randn(1, 3, 224, 224)
        heatmap = explainer.generate_heatmap(input_tensor)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_gradcam_explain_saves_files(self, sample_image, tmp_path):
        from backend.explainability.gradcam import GradCAMExplainer

        model = SimpleCNN()
        explainer = GradCAMExplainer(
            model=model,
            target_layers=[model.features[-2]],
            use_vit=False,
        )

        input_tensor = torch.randn(1, 3, 224, 224)
        result = explainer.explain(
            image=sample_image,
            input_tensor=input_tensor,
            output_dir=tmp_path,
            prefix="test_",
        )

        assert "heatmap_path" in result
        assert "overlay_path" in result
        assert (tmp_path / "test_heatmap.png").exists()
        assert (tmp_path / "test_overlay.png").exists()

    def test_gradcam_fallback_on_no_layers(self, sample_image, tmp_path):
        """GradCAM should produce fallback heatmap when no layers detected."""
        from backend.explainability.gradcam import GradCAMExplainer

        model = nn.Linear(10, 10)  # No conv layers
        explainer = GradCAMExplainer(
            model=model,
            use_vit=False,
        )

        input_tensor = torch.randn(1, 3, 224, 224)
        result = explainer.explain(
            image=sample_image,
            input_tensor=input_tensor,
            output_dir=tmp_path,
        )

        # Should still produce files (fallback)
        assert "heatmap_path" in result
        assert "overlay_path" in result


class TestAttentionRollout:
    """Tests for the Attention Rollout module."""

    def test_fallback_map_generation(self, sample_image, tmp_path):
        """Attention rollout should produce fallback when model has no attention."""
        from backend.explainability.attention import AttentionRollout

        model = nn.Linear(10, 10)
        rollout = AttentionRollout(model=model)

        result = rollout.explain(
            image=sample_image,
            input_tensor=torch.randn(1, 3, 224, 224),
            output_dir=tmp_path,
        )

        assert "attention_path" in result
        assert "attention_overlay_path" in result
        assert "attention_map" in result
        assert isinstance(result["attention_map"], np.ndarray)
