"""
Structured logging configuration using loguru.

Provides a pre-configured logger instance, a timing decorator
for performance monitoring, and structured JSON logging support.
"""

from __future__ import annotations

import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    json_logging: bool = False,
) -> None:
    """
    Configure the global logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for persistent logs.
        json_logging: If True, use structured JSON format.
    """
    # Remove default handler
    logger.remove()

    if json_logging:
        # Structured JSON logs for production
        logger.add(
            sys.stderr,
            level=level,
            format="{message}",
            serialize=True,
        )
    else:
        # Console handler with rich formatting
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    logger.info(f"Logger initialized — level={level}")


def timed(func: Callable) -> Callable:
    """
    Decorator that logs execution time of a function.

    Usage:
        @timed
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def log_system_info() -> None:
    """Log system information for diagnostics."""
    import platform

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Architecture: {platform.machine()}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(
            f"RAM: {mem.total / (1024**3):.1f} GB total, "
            f"{mem.available / (1024**3):.1f} GB available"
        )
        logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, "
                     f"{psutil.cpu_count(logical=True)} logical")
    except ImportError:
        pass

    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                logger.info(f"GPU {i}: {name} ({mem:.1f} GB)")
    except ImportError:
        pass
