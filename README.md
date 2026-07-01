# Medical Visual Question Answering System

A production-grade Medical VQA platform that combines **STLLaVA-Med** for medical reasoning, **Grad-CAM** for explainability, **Grounding DINO** for region proposal, and **SAM2** for precise segmentation.

## Architecture

```
User Image + Question
        │
        ▼
    FastAPI Server
        │
        ├──► STLLaVA-Med ──► Medical Answer
        │
        ├──► Grad-CAM ──► Attention Heatmap
        │
        └──► (Optional) Grounding DINO ──► SAM2 ──► Lesion Mask
```

## Quick Start

### Option 1: Direct Installation

```bash
# Clone the repository
git clone <repo-url>
cd Project

# Install dependencies
pip install -r requirements.txt

# Start the server (auto-downloads models on first run)
python backend/api/server.py
```

### Option 2: Docker

```bash
docker compose up
```

### Option 3: Linux Startup Script

```bash
bash startup.sh
```

**That's it.** The system automatically:
- Downloads STLLaVA-Med model weights from HuggingFace
- Installs the LLaVA package if missing
- Detects GPU/CPU and selects optimal quantization
- Creates all output directories
- Starts the FastAPI server on port 8000

## API Documentation

Once running, visit: **http://localhost:8000/docs**

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + model status |
| `/models/status` | GET | Detailed model availability |
| `/predict` | POST | Medical VQA (image upload) |
| `/predict/json` | POST | Medical VQA (base64 image) |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@chest_xray.png" \
  -F "question=What abnormality is visible?"
```

## Project Structure

```
Project/
├── backend/
│   ├── api/              # FastAPI server, routes, schemas
│   ├── models/           # STLLaVA-Med wrapper, loader, model manager
│   ├── explainability/   # Grad-CAM & attention rollout
│   ├── localization/     # Grounding DINO + SAM2
│   ├── utils/            # Image processing, logging, device, startup
│   ├── config/           # Centralized settings
│   ├── outputs/          # Generated answers, heatmaps, masks
│   └── predict.py        # Main orchestrator
├── scripts/              # Setup & utility scripts
├── tests/                # Unit & integration tests
├── Dockerfile            # Docker build
├── docker-compose.yml    # One-command deployment
├── startup.sh            # Linux bootstrap script
├── requirements.txt      # Python dependencies
└── .env.example          # Environment configuration
```

## Features

| Feature | Status |
|---------|--------|
| STLLaVA-Med Inference | ✅ Production |
| Automatic Model Downloads | ✅ Production |
| Grad-CAM Explainability | ✅ Production |
| Attention Rollout | ✅ Production |
| Grounding DINO Localization | ✅ Production |
| SAM2 Segmentation | ✅ Production |
| GPU Auto-Detection | ✅ Production |
| Memory-Aware Quantization | ✅ Production |
| Docker Deployment | ✅ Production |
| API Hardening | ✅ Production |
| Startup Validation | ✅ Production |
| Structured Logging | ✅ Production |

## GPU Requirements

| GPU Memory | Quantization | Performance |
|-----------|-------------|-------------|
| ≥ 40 GB (A100) | fp16 full | Fastest |
| ≥ 24 GB (RTX 3090/4090) | fp16 | Fast |
| ≥ 16 GB (T4/V100) | 8-bit | Good |
| ≥ 10 GB (RTX 3060) | 4-bit | Acceptable |
| CPU only | fp32 | Slow (development) |

## Google Colab

```python
# Run in a Colab cell:
!git clone <repo-url>
%cd Project
!pip install -r requirements.txt
!python backend/api/server.py
```

## Resources

- [STLLaVA-Med](https://github.com/heliossun/STLLaVA-Med)
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2](https://github.com/facebookresearch/sam2)
