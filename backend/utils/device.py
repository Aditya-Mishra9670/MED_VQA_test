"""
Device detection and memory management utilities.

Provides auto-detection of CUDA/MPS/CPU, memory monitoring,
and quantization recommendations for model placement decisions.
"""

from __future__ import annotations

from loguru import logger

try:
    import torch
except ImportError:
    torch = None


def get_device(preference: str = "auto") -> "torch.device":
    """
    Resolve device preference to a torch.device.

    Priority: CUDA → MPS → CPU

    Args:
        preference: One of 'auto', 'cuda', 'mps', 'cpu', 'cuda:0', etc.

    Returns:
        torch.device for model placement.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto-detected Apple MPS device")
            return torch.device("mps")
        logger.info("No GPU detected, using CPU")
        return torch.device("cpu")

    if preference.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            f"Requested {preference} but CUDA is not available. "
            f"Falling back to CPU."
        )
        return torch.device("cpu")

    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("Requested MPS but it is not available. Falling back to CPU.")
            return torch.device("cpu")

    return torch.device(preference)


def get_device_info() -> dict:
    """
    Gather device information for diagnostics.

    Returns:
        Dictionary with device name, CUDA availability, GPU memory, etc.
    """
    info = {
        "pytorch_version": torch.__version__ if torch else "not installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if torch else False
        ),
        "device_count": torch.cuda.device_count() if (torch and torch.cuda.is_available()) else 0,
        "current_device": None,
        "device_name": None,
        "total_memory_gb": None,
        "allocated_memory_gb": None,
        "reserved_memory_gb": None,
        "recommended_quantization": None,
    }

    if torch and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["current_device"] = idx
        info["device_name"] = torch.cuda.get_device_name(idx)
        total_mem = torch.cuda.get_device_properties(idx).total_mem
        info["total_memory_gb"] = round(total_mem / (1024**3), 2)
        info["allocated_memory_gb"] = round(
            torch.cuda.memory_allocated(idx) / (1024**3), 2
        )
        info["reserved_memory_gb"] = round(
            torch.cuda.memory_reserved(idx) / (1024**3), 2
        )
        info["recommended_quantization"] = get_quantization_recommendation(
            info["total_memory_gb"]
        )

    return info


def get_quantization_recommendation(gpu_memory_gb: float) -> str:
    """
    Recommend quantization strategy based on available GPU memory.

    Args:
        gpu_memory_gb: Total GPU memory in gigabytes.

    Returns:
        Recommended quantization strategy string.
    """
    if gpu_memory_gb >= 40:
        return "fp16 (full precision — ample memory)"
    elif gpu_memory_gb >= 24:
        return "fp16 (standard — sufficient memory)"
    elif gpu_memory_gb >= 16:
        return "8-bit quantization (limited memory)"
    elif gpu_memory_gb >= 10:
        return "4-bit quantization (tight memory)"
    else:
        return "4-bit quantization (very limited memory — may OOM)"


def get_available_memory_gb() -> float:
    """Get available GPU memory in GB. Returns 0 if no GPU."""
    if torch is None or not torch.cuda.is_available():
        return 0.0
    idx = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_mem
    allocated = torch.cuda.memory_allocated(idx)
    return round((total - allocated) / (1024**3), 2)


def clear_gpu_memory() -> None:
    """Free unused GPU memory."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        logger.debug("GPU memory cache cleared")


def check_memory_for_model(required_gb: float = 8.0) -> bool:
    """
    Check if enough GPU memory is available for a model.

    Args:
        required_gb: Minimum required GPU memory in GB.

    Returns:
        True if enough memory is available or running on CPU.
    """
    if torch is None or not torch.cuda.is_available():
        return True  # CPU mode, memory managed by OS

    available = get_available_memory_gb()
    if available < required_gb:
        logger.warning(
            f"Low GPU memory: {available:.1f}GB available, "
            f"{required_gb:.1f}GB recommended. Consider 4-bit quantization."
        )
        return False
    return True
