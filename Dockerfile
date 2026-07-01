# ============================================
# Medical VQA System — Dockerfile
# ============================================
# Multi-stage build with CUDA support
#
# Build: docker build -t medical-vqa .
# Run:   docker run --gpus all -p 8000:8000 medical-vqa

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# ── Install Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──
COPY backend/ backend/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pyproject.toml .
COPY .env.example .env

# Install the project
RUN pip install --no-cache-dir -e .

# ── Create output directories ──
RUN mkdir -p backend/outputs/answers \
             backend/outputs/heatmaps \
             backend/outputs/masks \
             checkpoints \
             model_cache \
             logs

# ── Expose port ──
EXPOSE 8000

# ── Health check ──
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ──
CMD ["python", "backend/api/server.py"]
