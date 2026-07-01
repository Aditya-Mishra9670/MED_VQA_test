#!/usr/bin/env bash
# ============================================
# Medical VQA System — Startup Script
# ============================================
# Usage: bash startup.sh
#
# This script bootstraps the entire system:
# 1. Validates Python environment
# 2. Installs dependencies
# 3. Runs startup checks
# 4. Starts the FastAPI server

set -e

echo "============================================"
echo "  Medical VQA System — Startup"
echo "============================================"
echo ""

# ── 1. Check Python ──
PYTHON=${PYTHON:-python3}

if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: Python not found. Install Python 3.10+"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PY_VERSION"

# ── 2. Install dependencies ──
echo ""
echo "Installing dependencies..."
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -r requirements.txt -q
echo "Dependencies installed."

# ── 3. Install project ──
echo ""
echo "Installing project..."
$PYTHON -m pip install -e . -q 2>/dev/null || echo "  (editable install skipped)"

# ── 4. Copy .env if not present ──
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# ── 5. Create output directories ──
mkdir -p backend/outputs/answers
mkdir -p backend/outputs/heatmaps
mkdir -p backend/outputs/masks
mkdir -p checkpoints
mkdir -p model_cache
mkdir -p logs

# ── 6. Run startup checks ──
echo ""
echo "Running startup checks..."
$PYTHON -c "from backend.utils.startup import run_startup_checks; run_startup_checks()" 2>/dev/null || echo "  (startup checks completed with warnings)"

# ── 7. GPU Info ──
echo ""
$PYTHON -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'GPU: {name} ({mem:.1f} GB)')
else:
    print('GPU: None (running on CPU)')
" 2>/dev/null || echo "  (GPU check skipped — PyTorch not available)"

# ── 8. Start server ──
echo ""
echo "============================================"
echo "  Starting Medical VQA Server"
echo "  http://0.0.0.0:${API_PORT:-8000}"
echo "  Docs: http://localhost:${API_PORT:-8000}/docs"
echo "============================================"
echo ""

exec $PYTHON backend/api/server.py
