"""
Centralized configuration for the Medical VQA System.

All settings are loaded from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = Project/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings with env variable support."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- STLLaVA-Med ----
    stllava_model_path: str = "ZachSun/stllava-med-7b"
    stllava_model_base: str = "liuhaotian/llava-v1.5-7b"

    # ---- Grounding DINO ----
    grounding_dino_checkpoint: str = "checkpoints/groundingdino_swint_ogc.pth"
    grounding_dino_config: str = (
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )

    # ---- SAM2 ----
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # ---- Inference ----
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ---- Model Management ----
    model_cache_dir: str = (
        str(Path("/kaggle/input/model_cache")) 
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ 
        else str(PROJECT_ROOT / "model_cache")
    )
    auto_download_models: bool = True

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = 50

    # ---- Output ----
    output_dir: str = str(PROJECT_ROOT / "backend" / "outputs")
    log_level: str = "INFO"
    log_file: str = ""

    # ---- Security ----
    allowed_mime_types: list[str] = [
        "image/png", "image/jpeg", "image/jpg", "image/bmp",
        "image/tiff", "image/webp", "image/dicom",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device string."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def effective_quantization(self) -> dict:
        """Determine effective quantization based on device and memory."""
        device = self.resolved_device
        if device == "cpu":
            return {"load_in_8bit": False, "load_in_4bit": False, "dtype": "float32"}

        if self.load_in_4bit:
            return {"load_in_8bit": False, "load_in_4bit": True, "dtype": "float16"}
        if self.load_in_8bit:
            return {"load_in_8bit": True, "load_in_4bit": False, "dtype": "float16"}

        # Auto-detect based on GPU memory
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                    if gpu_mem_gb < 12:
                        return {"load_in_8bit": False, "load_in_4bit": True, "dtype": "float16"}
                    elif gpu_mem_gb < 24:
                        return {"load_in_8bit": True, "load_in_4bit": False, "dtype": "float16"}
            except Exception:
                pass

        return {"load_in_8bit": self.load_in_8bit, "load_in_4bit": self.load_in_4bit, "dtype": "float16"}

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def answers_dir(self) -> Path:
        path = Path(self.output_dir) / "answers"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def heatmaps_dir(self) -> Path:
        path = Path(self.output_dir) / "heatmaps"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def masks_dir(self) -> Path:
        path = Path(self.output_dir) / "masks"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoints_dir(self) -> Path:
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            return Path("/kaggle/input/checkpoints")
        path = PROJECT_ROOT / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
