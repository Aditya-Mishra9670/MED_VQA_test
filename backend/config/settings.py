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
from dotenv import load_dotenv

# Project root = Project/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables into os.environ for external libraries (like huggingface_hub)
load_dotenv(PROJECT_ROOT / ".env")


def _get_model_cache_dir() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "/kaggle/working/model_cache"
    return str(PROJECT_ROOT / "model_cache")

def _get_checkpoints_dir() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "/kaggle/working/checkpoints"
    return str(PROJECT_ROOT / "checkpoints")

def _find_in_kaggle(dataset_name: str, target_file: str) -> str | None:
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        return None
    
    base = Path("/kaggle/input")
    if not base.exists():
        return None

    # Kaggle mounts inputs in /kaggle/input/.
    # Notebooks used as inputs might be named "STLLava01" or similar.
    # We search for the target file and check if the dataset_name is in the path.
    matches = []
    for p in base.rglob(target_file):
        if dataset_name.lower() in str(p).lower():
            return str(p)
        matches.append(p)
        
    # Fallback to the first match if any exist
    if matches:
        return str(matches[0])
        
    return None

def _get_stllava_path() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        found = _find_in_kaggle("stllava01", "config.json")
        if found:
            return str(Path(found).parent)
        return "/kaggle/input/notebooks/systemsuperadmin/stllava01"
    return "ZachSun/stllava-med-7b"

def _get_stllava_base() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        found = _find_in_kaggle("stllava02", "config.json")
        if found:
            return str(Path(found).parent)
        return "/kaggle/input/notebooks/systemsuperadmin/stllava02"
    return "liuhaotian/llava-v1.5-7b"

def _get_gdino_ckpt() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        found = _find_in_kaggle("groundingdino", "groundingdino_swint_ogc.pth")
        if found:
            return found
        return "/kaggle/input/notebooks/systemsuperadmin/groundingdino/groundingdino_swint_ogc.pth"
    return "checkpoints/groundingdino_swint_ogc.pth"

def _get_sam2_ckpt() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        found = _find_in_kaggle("sam2-model", "sam2.1_hiera_large.pt")
        if found:
            return found
        return "/kaggle/input/notebooks/systemsuperadmin/sam2-model/sam2.1_hiera_large.pt"
    return "checkpoints/sam2.1_hiera_large.pt"

class Settings(BaseSettings):
    """Application-wide settings with env variable support."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- STLLaVA-Med ----
    stllava_model_path: str = _get_stllava_path()
    stllava_model_base: str = _get_stllava_base()

    # ---- Grounding DINO ----
    grounding_dino_checkpoint: str = _get_gdino_ckpt()
    grounding_dino_config: str = (
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )

    # ---- SAM2 ----
    sam2_checkpoint: str = _get_sam2_ckpt()
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # ---- Inference ----
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ---- Model Management ----
    model_cache_dir: str = _get_model_cache_dir()
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
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if gpu_mem_gb < 24:
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
            return Path(_get_checkpoints_dir())
        path = PROJECT_ROOT / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
