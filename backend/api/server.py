"""
FastAPI application entry point for the Medical VQA System.

Configures:
- CORS middleware for cross-origin requests
- Static file serving for generated outputs
- Lifespan events for model loading on startup
- API route mounting
- Global exception handlers
- Startup validation

Run directly:
    python backend/api/server.py

Or with uvicorn:
    uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# ── Fix sys.path for direct script execution ──
# When running `python backend/api/server.py`, the Project root must be on sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.config.settings import get_settings
from backend.utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: Initialize logger, validate environment, optionally pre-load models.
    Shutdown: Clean up resources.
    """
    settings = get_settings()
    setup_logger(level=settings.log_level, log_file=settings.log_file or None)

    logger.info("=" * 60)
    logger.info("Medical VQA System — Starting up")
    logger.info(f"Device: {settings.resolved_device}")
    logger.info(f"Output dir: {settings.output_dir}")
    logger.info(f"Model cache: {settings.model_cache_dir}")
    logger.info(f"Auto-download: {settings.auto_download_models}")
    logger.info("=" * 60)

    # Run startup validation
    try:
        from backend.utils.startup import run_startup_checks
        run_startup_checks()
    except Exception as e:
        logger.warning(f"Startup checks encountered issues: {e}")

    # Ensure output directories exist
    settings.answers_dir
    settings.heatmaps_dir
    settings.masks_dir

    # Auto-download models if enabled
    if settings.auto_download_models:
        try:
            from backend.models.model_manager import ModelManager
            manager = ModelManager()
            manager.ensure_stllava_available()
            logger.info("Core model availability verified")
        except Exception as e:
            logger.warning(
                f"Model auto-download encountered issues: {e}. "
                f"Models will be downloaded on first request."
            )

    yield

    # Shutdown
    logger.info("Medical VQA System — Shutting down")
    try:
        from backend.models.loader import ModelLoader
        ModelLoader().unload_all()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Medical Visual Question Answering System",
        description=(
            "A modular Medical VQA platform combining STLLaVA-Med "
            "for medical reasoning, Grad-CAM for explainability, "
            "Grounding DINO for region proposal, and SAM2 for "
            "precise segmentation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Global exception handler ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": f"{type(exc).__name__}: {str(exc)}",
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.error(f"Runtime error: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "detail": str(exc),
            },
        )

    # ── CORS middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

    # ── Mount static files for serving generated outputs ──
    output_path = Path(settings.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/outputs",
        StaticFiles(directory=str(output_path)),
        name="outputs",
    )

    # ── Include API routes ──
    from backend.api.routes import router
    app.include_router(router)

    return app


# Application instance (used by: uvicorn backend.api.server:app)
app = create_app()


if __name__ == "__main__":
    """
    Direct execution entry point.
    Allows: python backend/api/server.py
    """
    import uvicorn

    settings = get_settings()

    print()
    print("=" * 60)
    print("  Medical VQA System")
    print(f"  Starting on http://{settings.api_host}:{settings.api_port}")
    print(f"  Device: {settings.resolved_device}")
    print(f"  Docs: http://localhost:{settings.api_port}/docs")
    print("=" * 60)
    print()

    uvicorn.run(
        "backend.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
        workers=1,  # Single worker for GPU model sharing
    )
