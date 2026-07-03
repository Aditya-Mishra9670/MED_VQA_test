Viewed README.md:1-141

Based on the system architecture and the models used in this project, here are the total hardware requirements (VRAM, Disk Space, and System RAM) to run the backend successfully.

### 1. GPU VRAM Requirements
The primary bottleneck is the **STLLaVA-Med 7B** model. Depending on how much you compress (quantize) the model, your VRAM needs will change:

*   **Optimal / Fastest (FP16):** `≥ 24 GB VRAM` *(e.g., RTX 3090, RTX 4090, A5000)*
*   **Good / Balanced (8-bit Quantization):** `≥ 16 GB VRAM` *(e.g., NVIDIA T4, V100, RTX 4080)*
*   **Minimum Acceptable (4-bit Quantization):** `≥ 10 GB VRAM` *(e.g., RTX 3060 12GB)*
*   *Note: If you enable SAM2 and GroundingDINO for localization/segmentation, they will consume an additional ~2GB of VRAM.*

### 2. Disk Space Requirements
You should have at least **40 GB to 50 GB of free disk space**. Here is the breakdown:

*   **STLLaVA-Med Weights:** ~14.1 GB
*   **LLaVA Base Model (Cached config/weights):** ~14.0 GB
*   **SAM2 Checkpoint (Hiera Large):** ~1.2 GB
*   **Grounding DINO Checkpoint:** ~400 MB
*   **Python Dependencies:** ~5 GB *(PyTorch, CUDA toolkits, transformers, etc.)*
*   **Docker Image (if using Docker):** The final built image is usually around ~20 GB because it packages Ubuntu, CUDA dev libraries, and PyTorch.

### 3. System RAM (CPU Memory)
When the application first starts, it loads the ~14GB model weights from your disk into System RAM before moving them to the GPU. 
*   **Recommended:** `32 GB RAM`
*   **Minimum:** `16 GB RAM` *(If you have exactly 16GB, you may need to rely heavily on your operating system's swap file during the initial model load, which can be slow).*

### 4. Operating System
*   **Linux (Ubuntu 20.04/22.04) or WSL2 on Windows:** Highly recommended because compiling packages like `GroundingDINO` natively on Windows often fails due to missing C++ Build Tools and CUDA compiler mismatches. 
*   **Docker:** If using Docker (via `docker compose up`), the host OS doesn't matter as long as you have the NVIDIA Container Toolkit installed.