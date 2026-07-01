# Medical VQA Deployment & Testing Guide

This guide describes how to deploy the Medical VQA backend to a GPU-enabled environment (Google Colab or RunPod) and host the frontend on Vercel or locally for testing.

---

## Part 1: Backend Deployment

Since STLLaVA-Med (7B parameters) and SAM2 require high GPU VRAM (minimum 15GB to 24GB recommended), hosting the backend on CPU-only standard servers is not viable. You have two main options:

### Option A: Google Colab (Free / Low Cost) — *easiest for testing*

Google Colab provides a free NVIDIA T4 GPU (15GB VRAM) or paid L4/A100 GPUs. We can expose the FastAPI backend running in Colab via a secure tunnel using **ngrok**.

#### Step 1: Open Colab & Upload Code
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and set the runtime to **T4 GPU** (Runtime > Change runtime type > T4 GPU).
3. Connect your Google Drive or upload the project folder (especially the `backend` directory and `colab_setup.py`).

#### Step 2: Run Setup Script
In a Colab cell, execute:
```python
# Upload colab_setup.py or paste its contents and run:
import colab_setup
colab_setup.setup_colab()
```

#### Step 3: Start the Backend with Ngrok Tunneling
Create an account on [ngrok.com](https://ngrok.com/) to get a free authtoken. In a new cell, run:
```python
from pyngrok import ngrok
import os

# Set your Ngrok Authtoken
NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN"
ngrok.set_auth_token(NGROK_TOKEN)

# Open tunnel on port 8000
public_url = ngrok.connect(8000)
print(f"Backend Public URL: {public_url}")

# Run FastAPI app
!uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```
Keep this cell running. Copy the `Backend Public URL` (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app`).

---

### Option B: RunPod / Vast.ai (Paid Cloud GPU) — *most robust*

For constant availability and full environment control, you can rent a GPU instance (RTX 3090, 4090, or A5000 with 24GB VRAM) for roughly $0.20 to $0.40 per hour.

1. **Launch Pod**: Choose a **PyTorch** template on RunPod.
2. **Access SSH/Jupyter**: Connect to your pod.
3. **Clone Repo / Upload Code**: Upload the workspace code.
4. **Install backend requirements**:
   ```bash
   pip install -r backend/requirements.txt
   ```
5. **Start server**:
   ```bash
   uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
   ```
6. **Expose Ports**: RunPod allows you to expose port 8000 using their HTTP Port mapping. Copy the public address provided by RunPod.

---

## Part 2: Frontend Deployment & Testing

Once you have your `Backend Public URL` from Colab or RunPod, you can link the frontend.

### Option A: Local Testing (pointing to Remote Backend)

1. Navigate to the `frontend` folder on your local machine.
2. Create or edit `frontend/.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=https://xxxx-xx-xx-xx.ngrok-free.app
   ```
   *(Replace with your actual public URL from Colab/RunPod)*
3. Run the development build:
   ```bash
   npm run dev
   ```
4. Access [http://localhost:3000](http://localhost:3000) and start testing.

### Option B: Production Deployment on Vercel (Free Hosting)

Vercel is the default choice for deploying Next.js applications.

1. **Prepare Code**: Commit your workspace to a GitHub repository.
2. **Import to Vercel**: 
   - Go to [Vercel](https://vercel.com/) and link your GitHub account.
   - Click **Add New** > **Project** and select your repository.
3. **Configure Build Settings**:
   - **Root Directory**: `frontend`
   - **Environment Variables**: Add a new environment variable:
     - **Key**: `NEXT_PUBLIC_API_URL`
     - **Value**: `https://xxxx-xx-xx-xx.ngrok-free.app` (your backend URL)
4. **Deploy**: Click **Deploy**. Vercel will build and host your app on a public `vercel.app` URL.

---

## Part 3: Verifying the Connection

1. Open your deployed frontend (or `localhost:3000`).
2. Navigate to the **System Status** page (`/status`).
3. If the connection is successful:
   - The status badge will show `Operational`.
   - The Active Device (e.g., `cuda`) and loaded models will populate.
4. Go to **Analyze**, upload an image, and submit a question to run an end-to-end inference test.
