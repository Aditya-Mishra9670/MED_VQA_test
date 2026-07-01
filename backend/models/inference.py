"""
High-level inference pipeline for Medical VQA.

Orchestrates STLLaVA-Med inference with result caching.
This module is the primary interface used by predict.py and the API.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger

from backend.config.settings import get_settings
from backend.models.loader import ModelLoader
from backend.utils.logger import timed


@dataclass
class VQAResult:
    """Result container for a single VQA prediction."""

    answer: str
    question: str
    inference_time_seconds: float
    model_name: str = "stllava-med-7b"
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "question": self.question,
            "model_name": self.model_name,
            "inference_time_seconds": round(self.inference_time_seconds, 3),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class MedicalVQAInference:
    """
    Medical VQA inference engine.

    Wraps model loading and inference with:
    - Lazy model initialization
    - Result caching (optional)
    - Performance timing
    - Answer persistence
    """

    def __init__(self, enable_cache: bool = True):
        self._loader = ModelLoader()
        self._cache: dict[str, VQAResult] = {}
        self._enable_cache = enable_cache
        self._settings = get_settings()

    def _compute_cache_key(self, image: Image.Image, question: str) -> str:
        """Generate a hash key from image content + question text."""
        import numpy as np
        img_array = np.array(image.resize((224, 224)))
        img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        q_hash = hashlib.md5(question.encode()).hexdigest()
        return f"{img_hash}_{q_hash}"

    @timed
    def predict(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> VQAResult:
        """
        Run medical VQA inference.

        Args:
            image: PIL Image of a medical scan.
            question: Clinical question about the image.
            max_new_tokens: Override max generation length.
            temperature: Override sampling temperature.

        Returns:
            VQAResult with the generated answer and metadata.
        """
        # Check cache
        if self._enable_cache:
            cache_key = self._compute_cache_key(image, question)
            if cache_key in self._cache:
                logger.info(f"Cache hit for question: {question[:50]}...")
                return self._cache[cache_key]

        # Run inference
        start = time.perf_counter()

        answer = self._loader.stllava.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        elapsed = time.perf_counter() - start

        result = VQAResult(
            answer=answer,
            question=question,
            inference_time_seconds=elapsed,
            metadata={
                "device": self._settings.resolved_device,
                "max_new_tokens": max_new_tokens or self._settings.max_new_tokens,
                "temperature": temperature or self._settings.temperature,
            },
        )

        # Cache result
        if self._enable_cache:
            self._cache[cache_key] = result

        # Persist answer
        self._save_answer(result)

        return result

    def _save_answer(self, result: VQAResult) -> None:
        """Save answer to the outputs directory."""
        try:
            answers_dir = self._settings.answers_dir
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"answer_{timestamp}.json"
            filepath = answers_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Answer saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save answer: {e}")

    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self._cache.clear()
        logger.info("Inference cache cleared.")

    @property
    def model_status(self) -> dict:
        """Get model loading status."""
        return self._loader.status()
