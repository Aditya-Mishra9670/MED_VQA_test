"""
Startup validation for the Medical VQA System.

Runs comprehensive checks on application startup to ensure
all dependencies, models, directories, and hardware are properly
configured. Produces human-readable error messages instead of
cryptic stack traces.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from loguru import logger


def run_startup_checks() -> dict:
    """
    Run all startup validation checks.

    Returns:
        Dict with check results.
    """
    results = {}

    results["python"] = _check_python_version()
    results["dependencies"] = _check_dependencies()
    results["torch"] = _check_torch()
    results["gpu"] = _check_gpu()
    results["directories"] = _check_directories()

    # Log summary
    passed = sum(1 for v in results.values() if v.get("ok"))
    total = len(results)
    logger.info(f"Startup checks: {passed}/{total} passed")

    for name, result in results.items():
        if not result.get("ok"):
            logger.warning(f"  ⚠ {name}: {result.get('message', 'unknown issue')}")
        else:
            logger.info(f"  ✓ {name}: {result.get('message', 'ok')}")

    return results


def _check_python_version() -> dict:
    """Check Python version compatibility."""
    version = sys.version_info
    if version >= (3, 10):
        return {
            "ok": True,
            "message": f"Python {version.major}.{version.minor}.{version.micro}",
        }
    return {
        "ok": False,
        "message": (
            f"Python {version.major}.{version.minor} detected. "
            f"Python 3.10+ is required."
        ),
    }


def _check_dependencies() -> dict:
    """Check that critical Python packages are importable."""
    required = {
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "pydantic": "Data validation",
        "PIL": "Image processing (Pillow)",
        "cv2": "Computer vision (OpenCV)",
        "numpy": "Numerical computing",
        "loguru": "Logging",
    }

    optional = {
        "huggingface_hub": "Model downloads",
        "transformers": "ML models",
        "torch": "PyTorch",
        "pytorch_grad_cam": "Grad-CAM explainability",
    }

    missing_required = []
    missing_optional = []

    for pkg, desc in required.items():
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_required.append(f"{pkg} ({desc})")

    for pkg, desc in optional.items():
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_optional.append(f"{pkg} ({desc})")

    if missing_required:
        return {
            "ok": False,
            "message": (
                f"Missing required packages: {', '.join(missing_required)}. "
                f"Run: pip install -r requirements.txt"
            ),
        }

    msg = "All required packages available"
    if missing_optional:
        msg += f". Optional missing: {', '.join(missing_optional)}"

    return {"ok": True, "message": msg}


def _check_torch() -> dict:
    """Check PyTorch installation and version."""
    try:
        import torch

        info = {
            "ok": True,
            "message": f"PyTorch {torch.__version__}",
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }

        if hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()

        return info

    except ImportError:
        return {
            "ok": False,
            "message": (
                "PyTorch not installed. Install with: "
                "pip install torch torchvision"
            ),
        }


def _check_gpu() -> dict:
    """Check GPU availability and memory."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024 ** 3)

            message = f"CUDA GPU: {device_name} ({mem_gb:.1f} GB)"

            if mem_gb < 8:
                message += " — WARNING: Low VRAM, use 4-bit quantization"
            elif mem_gb < 16:
                message += " — Recommended: 8-bit quantization"

            return {"ok": True, "message": message, "memory_gb": mem_gb}

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "ok": True,
                "message": "Apple MPS GPU available. Some features may be limited.",
            }

        else:
            return {
                "ok": True,
                "message": (
                    "No GPU detected — running on CPU. "
                    "Inference will be slow. Consider using Google Colab or "
                    "a cloud GPU instance."
                ),
            }

    except ImportError:
        return {"ok": False, "message": "PyTorch not installed, cannot check GPU"}


def _check_directories() -> dict:
    """Check and create required directories."""
    try:
        from backend.config.settings import get_settings

        settings = get_settings()

        dirs = [
            settings.answers_dir,
            settings.heatmaps_dir,
            settings.masks_dir,
            Path(settings.model_cache_dir),
            settings.checkpoints_dir,
        ]

        created = []
        for d in dirs:
            d = Path(d)
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created.append(str(d))

        msg = "All directories ready"
        if created:
            msg += f". Created: {len(created)} directories"

        return {"ok": True, "message": msg}

    except Exception as e:
        return {"ok": False, "message": f"Directory check failed: {e}"}
