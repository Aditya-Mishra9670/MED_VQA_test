"""
Model loader with singleton pattern and lazy initialization.

Manages loading and caching of all models used in the pipeline:
- STLLaVA-Med (medical VQA)
- Grounding DINO (localization)
- SAM2 (segmentation)

Models are loaded on first access and cached for subsequent requests.
Integrates with ModelManager for automatic downloads.
"""

from __future__ import annotations

import threading
from typing import Optional

from loguru import logger

from backend.config.settings import Settings, get_settings
from backend.models.stllava import STLLaVAMed, STLLaVAConfig


class ModelLoader:
    """
    Thread-safe lazy model loader with singleton pattern.

    Loads models on first access and keeps them in memory.
    Supports device placement and optional quantization.
    Integrates with ModelManager to auto-download models before loading.
    """

    _instance: Optional[ModelLoader] = None
    _lock = threading.Lock()
    _stllava: Optional[STLLaVAMed] = None
    _grounding_dino = None
    _sam2 = None

    def __new__(cls) -> ModelLoader:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._settings = get_settings()

    @property
    def stllava(self) -> STLLaVAMed:
        """Get or load the STLLaVA-Med model."""
        if self._stllava is None:
            with self._lock:
                if self._stllava is None:
                    logger.info("Initializing STLLaVA-Med model...")

                    # Ensure model is downloaded
                    try:
                        from backend.models.model_manager import ModelManager
                        manager = ModelManager()
                        manager.ensure_stllava_available()
                    except Exception as e:
                        logger.warning(f"Model pre-download check failed: {e}")

                    quant = self._settings.effective_quantization
                    config = STLLaVAConfig(
                        model_path=self._settings.stllava_model_path,
                        model_base=self._settings.stllava_model_base,
                        device=self._settings.resolved_device,
                        max_new_tokens=self._settings.max_new_tokens,
                        temperature=self._settings.temperature,
                        load_in_8bit=quant["load_in_8bit"],
                        load_in_4bit=quant["load_in_4bit"],
                    )
                    self._stllava = STLLaVAMed(config)
                    self._stllava.load()
        return self._stllava

    @property
    def grounding_dino(self):
        """Get or load the Grounding DINO model."""
        if self._grounding_dino is None:
            with self._lock:
                if self._grounding_dino is None:
                    logger.info("Initializing Grounding DINO model...")

                    # Ensure checkpoint is downloaded
                    checkpoint = self._settings.grounding_dino_checkpoint
                    try:
                        from backend.models.model_manager import ModelManager
                        manager = ModelManager()
                        result = manager.ensure_grounding_dino_available()
                        if result:
                            checkpoint = result
                    except Exception as e:
                        logger.warning(f"Grounding DINO download check failed: {e}")

                    from backend.localization.grounding_dino import GroundingDINOWrapper
                    self._grounding_dino = GroundingDINOWrapper(
                        config_path=self._settings.grounding_dino_config,
                        checkpoint_path=checkpoint,
                        device=self._settings.resolved_device,
                    )
                    self._grounding_dino.load()
        return self._grounding_dino

    @property
    def sam2(self):
        """Get or load the SAM2 model."""
        if self._sam2 is None:
            with self._lock:
                if self._sam2 is None:
                    logger.info("Initializing SAM2 model...")

                    # Ensure checkpoint is downloaded
                    checkpoint = self._settings.sam2_checkpoint
                    try:
                        from backend.models.model_manager import ModelManager
                        manager = ModelManager()
                        result = manager.ensure_sam2_available()
                        if result:
                            checkpoint = result
                    except Exception as e:
                        logger.warning(f"SAM2 download check failed: {e}")

                    from backend.localization.sam2 import SAM2Wrapper
                    self._sam2 = SAM2Wrapper(
                        checkpoint_path=checkpoint,
                        config_path=self._settings.sam2_config,
                        device=self._settings.resolved_device,
                    )
                    self._sam2.load()
        return self._sam2

    def load_all(self) -> None:
        """Pre-load all models (useful for server startup)."""
        logger.info("Pre-loading all models...")
        _ = self.stllava
        logger.info("All core models loaded.")

    def load_localization(self) -> None:
        """Load optional localization models (Grounding DINO + SAM2)."""
        logger.info("Loading localization models...")
        _ = self.grounding_dino
        _ = self.sam2
        logger.info("Localization models loaded.")

    def unload_all(self) -> None:
        """Release all models from memory."""
        with self._lock:
            self.__class__._stllava = None
            self.__class__._grounding_dino = None
            self.__class__._sam2 = None
            self.__class__._instance = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("All models unloaded.")

    def status(self) -> dict:
        """Get loading status of all models."""
        return {
            "stllava_loaded": self._stllava is not None and self._stllava.is_loaded,
            "grounding_dino_loaded": self._grounding_dino is not None,
            "sam2_loaded": self._sam2 is not None,
            "device": self._settings.resolved_device,
            "quantization": self._settings.effective_quantization,
        }
