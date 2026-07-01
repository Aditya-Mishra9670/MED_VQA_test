"""
Model Management System for the Medical VQA Pipeline.

Handles automatic downloading, validation, and caching of all model
checkpoints required by the system:
- STLLaVA-Med (medical VQA)
- Grounding DINO (localization)
- SAM2 (segmentation)
- LLaVA base model

Uses HuggingFace Hub for downloads with resume support and integrity checks.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.config.settings import get_settings, PROJECT_ROOT


# ── Model Registry ──
MODELS = {
    "stllava": {
        "repo_id": "ZachSun/stllava-med-7b",
        "description": "STLLaVA-Med 7B — Medical VQA model",
        "type": "huggingface_snapshot",
    },
    "llava_base": {
        "repo_id": "liuhaotian/llava-v1.5-7b",
        "description": "LLaVA v1.5 7B — Base model for STLLaVA-Med",
        "type": "huggingface_snapshot",
    },
    "grounding_dino": {
        "repo_id": "ShilongLiu/GroundingDINO",
        "filename": "groundingdino_swint_ogc.pth",
        "description": "Grounding DINO SwinT — Object localization",
        "type": "huggingface_file",
        "fallback_url": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        ),
    },
    "sam2": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
        "description": "SAM2.1 Hiera Large — Segmentation",
        "type": "huggingface_file",
        "subfolder": "",
    },
}


class ModelManager:
    """
    Manages model downloads, validation, and caching.

    Features:
    - Auto-download from HuggingFace Hub
    - Resume interrupted downloads
    - Integrity validation
    - Offline reuse from HF cache
    - Retry failed downloads
    - LLaVA package auto-installation
    """

    def __init__(self):
        self._settings = get_settings()
        self._cache_dir = Path(self._settings.model_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir = self._settings.checkpoints_dir

    # ── Public API ──

    def check_models(self) -> dict:
        """
        Check availability of all models.

        Returns:
            Dict mapping model name to availability status.
        """
        status = {}
        status["stllava"] = self._check_hf_model("ZachSun/stllava-med-7b")
        status["llava_base"] = self._check_hf_model("liuhaotian/llava-v1.5-7b")
        status["llava_package"] = self._check_llava_package()
        status["grounding_dino_checkpoint"] = self._check_checkpoint(
            self._checkpoints_dir / "groundingdino_swint_ogc.pth"
        )
        status["grounding_dino_package"] = self._check_package("groundingdino")
        status["sam2_checkpoint"] = self._check_checkpoint(
            self._checkpoints_dir / "sam2.1_hiera_large.pt"
        )
        status["sam2_package"] = self._check_package("sam2")
        return status

    def download_models(self, include_localization: bool = False) -> None:
        """
        Download all required models.

        Args:
            include_localization: Also download Grounding DINO and SAM2.
        """
        logger.info("Starting model download process...")

        # 1. Ensure LLaVA package is available
        self.ensure_llava_package()

        # 2. Download STLLaVA-Med and Base weights
        self.ensure_stllava_available()
        self.ensure_llava_base_available()

        # 3. Optional: localization models
        if include_localization:
            self.ensure_grounding_dino_available()
            self.ensure_sam2_available()

        logger.info("Model download process complete.")

    def validate_models(self) -> dict:
        """
        Validate all downloaded models.

        Returns:
            Dict with validation results for each model.
        """
        results = {}

        # Validate STLLaVA
        try:
            stllava_path = self._get_hf_cache_path("ZachSun/stllava-med-7b")
            if stllava_path and stllava_path.exists():
                has_config = (stllava_path / "config.json").exists() or any(
                    stllava_path.glob("*.json")
                )
                results["stllava"] = {
                    "valid": has_config,
                    "path": str(stllava_path),
                }
            else:
                results["stllava"] = {"valid": False, "path": None}
        except Exception as e:
            results["stllava"] = {"valid": False, "error": str(e)}

        # Validate checkpoints
        for name, filename in [
            ("grounding_dino", "groundingdino_swint_ogc.pth"),
            ("sam2", "sam2.1_hiera_large.pt"),
        ]:
            path = self._checkpoints_dir / filename
            results[name] = {
                "valid": path.exists() and path.stat().st_size > 1024,
                "path": str(path) if path.exists() else None,
            }

        return results

    # ── STLLaVA-Med ──

    def ensure_stllava_available(self) -> str:
        """
        Ensure STLLaVA-Med model is downloaded and available.

        Returns:
            Path to the model directory or HuggingFace model ID.
        """
        repo_id = self._settings.stllava_model_path

        # If it's a local path that exists, use it directly
        if Path(repo_id).exists():
            logger.info(f"Using local STLLaVA-Med model: {repo_id}")
            return repo_id

        # Try to download/verify from HuggingFace
        logger.info(f"Ensuring STLLaVA-Med model '{repo_id}' is available...")
        try:
            path = self._download_hf_snapshot(repo_id)
            logger.info(f"STLLaVA-Med available at: {path}")
            return str(path)
        except Exception as e:
            logger.warning(
                f"Could not pre-download STLLaVA-Med: {e}. "
                f"The transformers library will download it on first use."
            )
            return repo_id

    def ensure_llava_base_available(self) -> str:
        """Ensure the LLaVA base model is downloaded."""
        repo_id = self._settings.stllava_model_base

        if Path(repo_id).exists():
            return repo_id

        logger.info(f"Ensuring LLaVA base model '{repo_id}' is available...")
        try:
            path = self._download_hf_snapshot(repo_id)
            logger.info(f"LLaVA base model available at: {path}")
            return str(path)
        except Exception as e:
            logger.warning(f"Could not pre-download LLaVA base: {e}")
            return repo_id

    # ── LLaVA Package ──

    def ensure_llava_package(self) -> bool:
        """
        Ensure the LLaVA Python package is installed.

        The STLLaVA-Med model requires the `llava` package which must
        be installed from source. This method auto-installs it.

        Returns:
            True if package is available.
        """
        if self._check_llava_package():
            logger.info("LLaVA package already installed.")
            return True

        logger.info("LLaVA package not found. Attempting auto-installation...")

        # Try installing from the STLLaVA-Med repo
        install_sources = [
            "git+https://github.com/heliossun/STLLaVA-Med.git",
        ]

        for source in install_sources:
            try:
                logger.info(f"Attempting to install LLaVA from: {source}")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", source],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    # Verify installation
                    importlib.invalidate_caches()
                    try:
                        import llava  # noqa: F401
                        logger.info(f"LLaVA package installed successfully from {source}")
                        return True
                    except ImportError:
                        continue
                else:
                    logger.debug(f"pip install failed: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout installing from {source}")
            except Exception as e:
                logger.debug(f"Install attempt failed: {e}")

        logger.warning(
            "Could not auto-install LLaVA package. "
            "For best results, install manually: "
            "pip install git+https://github.com/heliossun/STLLaVA-Med.git"
        )
        return False

    # ── Grounding DINO ──

    def ensure_grounding_dino_available(self) -> Optional[str]:
        """
        Ensure Grounding DINO checkpoint is downloaded.

        Returns:
            Path to the checkpoint file, or None if unavailable.
        """
        checkpoint_path = self._checkpoints_dir / "groundingdino_swint_ogc.pth"

        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1024:
            logger.info(f"Grounding DINO checkpoint found: {checkpoint_path}")
            return str(checkpoint_path)

        logger.info("Downloading Grounding DINO checkpoint...")
        try:
            path = self._download_hf_file(
                repo_id="ShilongLiu/GroundingDINO",
                filename="groundingdino_swint_ogc.pth",
                local_path=checkpoint_path,
            )
            logger.info(f"Grounding DINO checkpoint saved to: {path}")
            return str(path)
        except Exception as e:
            logger.warning(f"Failed to download Grounding DINO: {e}")

            # Try fallback URL
            try:
                fallback_url = MODELS["grounding_dino"]["fallback_url"]
                self._download_url(fallback_url, checkpoint_path)
                logger.info(f"Grounding DINO downloaded from fallback URL")
                return str(checkpoint_path)
            except Exception as e2:
                logger.error(f"All download attempts for Grounding DINO failed: {e2}")
                return None

    # ── SAM2 ──

    def ensure_sam2_available(self) -> Optional[str]:
        """
        Ensure SAM2 checkpoint is downloaded.

        Returns:
            Path to the checkpoint file, or None if unavailable.
        """
        checkpoint_path = self._checkpoints_dir / "sam2.1_hiera_large.pt"

        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1024:
            logger.info(f"SAM2 checkpoint found: {checkpoint_path}")
            return str(checkpoint_path)

        logger.info("Downloading SAM2 checkpoint...")
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id="facebook/sam2.1-hiera-large",
                filename="sam2.1_hiera_large.pt",
                local_dir=str(self._checkpoints_dir),
                resume_download=True,
            )

            # If downloaded to a different location, copy to expected path
            downloaded = Path(downloaded_path)
            if downloaded != checkpoint_path and downloaded.exists():
                import shutil
                shutil.copy2(str(downloaded), str(checkpoint_path))

            logger.info(f"SAM2 checkpoint saved to: {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to download SAM2: {e}")
            return None

    # ── Private Helpers ──

    def _check_hf_model(self, repo_id: str) -> bool:
        """Check if a HuggingFace model is cached locally."""
        try:
            from huggingface_hub import try_to_load_from_cache, HfFileSystem

            # Check if any files are cached
            cache_path = self._get_hf_cache_path(repo_id)
            return cache_path is not None and cache_path.exists()
        except Exception:
            return False

    def _check_checkpoint(self, path: Path) -> bool:
        """Check if a checkpoint file exists and has reasonable size."""
        return path.exists() and path.stat().st_size > 1024

    def _check_llava_package(self) -> bool:
        """Check if the LLaVA package is importable."""
        try:
            import transformers
            from unittest.mock import patch
            orig_register_config = transformers.AutoConfig.register
            orig_register_model = transformers.AutoModelForCausalLM.register

            def patched_register_config(cls, model_type, config_class, **kwargs):
                kwargs['exist_ok'] = True
                return orig_register_config(model_type, config_class, **kwargs)

            def patched_register_model(cls, config_class, model_class, **kwargs):
                kwargs['exist_ok'] = True
                return orig_register_model(config_class, model_class, **kwargs)

            # Apply robust backward compatibility patches for deleted transformers functions
            from backend.utils.transformers_patch import apply_transformers_patches
            apply_transformers_patches()

            with patch.object(transformers.AutoConfig, 'register', classmethod(patched_register_config)), \
                 patch.object(transformers.AutoModelForCausalLM, 'register', classmethod(patched_register_model)):
                import llava  # noqa: F401
            return True
        except Exception:
            return False

    def _check_package(self, package_name: str) -> bool:
        """Check if a Python package is importable."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def _download_hf_snapshot(self, repo_id: str) -> Path:
        """
        Download a full HuggingFace model repository.

        Uses snapshot_download with resume support.
        """
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(self._cache_dir),
            resume_download=True,
        )
        return Path(path)

    def _download_hf_file(
        self,
        repo_id: str,
        filename: str,
        local_path: Path,
    ) -> Path:
        """Download a single file from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_path.parent),
            resume_download=True,
        )

        downloaded = Path(downloaded_path)
        if downloaded != local_path and downloaded.exists():
            import shutil
            shutil.copy2(str(downloaded), str(local_path))

        return local_path

    def _download_url(self, url: str, dest: Path, retries: int = 3) -> Path:
        """
        Download a file from a URL with retry support.

        Args:
            url: URL to download from.
            dest: Destination file path.
            retries: Number of retry attempts.

        Returns:
            Path to downloaded file.
        """
        import requests

        dest.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{retries})...")
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                if total_size > 0 and downloaded < total_size:
                    raise RuntimeError(
                        f"Incomplete download: {downloaded}/{total_size} bytes"
                    )

                logger.info(f"Download complete: {dest} ({downloaded} bytes)")
                return dest

            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    continue
                raise

        raise RuntimeError(f"Failed to download {url} after {retries} attempts")

    def _get_hf_cache_path(self, repo_id: str) -> Optional[Path]:
        """Get the local cache path for a HuggingFace model."""
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(str(self._cache_dir))
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    # Return the latest revision path
                    for revision in repo.revisions:
                        return Path(revision.snapshot_path)
            return None
        except Exception:
            # Try manual path construction
            safe_name = repo_id.replace("/", "--")
            model_path = self._cache_dir / f"models--{safe_name}"
            if model_path.exists():
                snapshots = model_path / "snapshots"
                if snapshots.exists():
                    revisions = sorted(snapshots.iterdir())
                    if revisions:
                        return revisions[-1]
            return None
