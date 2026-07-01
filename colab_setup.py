"""
Enhanced setup script for Medical VQA on Google Colab.
Installs all dependencies and bootstraps the system.

Usage in Colab:
    !python colab_setup.py
"""
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def setup_colab():
    print("===== Medical VQA Colab Setup =====")

    # Verify GPU
    try:
        run_cmd("nvidia-smi")
    except Exception:
        print("WARNING: No GPU detected. Inference will be slow.")

    print("\nInstalling core requirements...")
    run_cmd(f"{sys.executable} -m pip install -q -r requirements.txt")

    print("\nInstalling quantization support...")
    run_cmd(f"{sys.executable} -m pip install -q bitsandbytes")

    print("\nInstalling STLLaVA-Med LLaVA fork...")
    try:
        run_cmd(
            f"{sys.executable} -m pip install -q --no-deps "
            "git+https://github.com/heliossun/STLLaVA-Med.git"
        )
    except Exception:
        print("WARNING: STLLaVA auto-install failed.")

    print("\nInstalling ngrok for tunneling...")
    run_cmd(f"{sys.executable} -m pip install -q pyngrok")

    print("\nInstalling optional localization dependencies...")

    # Install GroundingDINO
    print("\n[1/2] Installing GroundingDINO...")
    try:
        run_cmd(f"{sys.executable} -m pip install -q setuptools wheel")
        # Clone and build with CUDA env vars set at the shell level
        run_cmd(
            "cd /tmp && rm -rf GroundingDINO && "
            "git clone --quiet https://github.com/IDEA-Research/GroundingDINO.git && "
            "cd GroundingDINO && "
            "BUILD_WITH_CUDA=True CUDA_HOME=/usr/local/cuda "
            f"{sys.executable} -m pip install -q -e ."
        )
        print("GroundingDINO installed successfully.")
    except Exception:
        print("WARNING: GroundingDINO source build failed. Trying PyPI fallback...")
        try:
            run_cmd(f"{sys.executable} -m pip install -q groundingdino-py")
            print("GroundingDINO installed from PyPI.")
        except Exception:
            print("WARNING: GroundingDINO installation failed. Localization will be unavailable.")

    # Install SAM2
    print("\n[2/2] Installing SAM2...")
    try:
        run_cmd(f"{sys.executable} -m pip install -q git+https://github.com/facebookresearch/sam2.git")
        print("SAM2 installed successfully.")
    except Exception as e:
        print(f"WARNING: SAM2 installation failed: {e}. Segmentation will be unavailable.")

    print("\nRunning startup checks...")
    try:
        from backend.utils.startup import run_startup_checks
        run_startup_checks()
    except Exception as e:
        print(f"Startup checks encountered issues: {e}")

    print("\nPre-downloading all models (STLLaVA, Base, GroundingDINO, SAM2)...")
    try:
        from backend.models.model_manager import ModelManager
        manager = ModelManager()
        manager.download_models(include_localization=True)
        print("All models ready.")
    except Exception as e:
        print(f"Model pre-download skipped: {e}")

    print("\n===== Setup Complete =====")
    print("\nTo start the server, run:")
    print("  python backend/api/server.py")
    print("\nOr with ngrok tunnel:")
    print("  from pyngrok import ngrok")
    print("  ngrok.set_auth_token('YOUR_TOKEN')")
    print("  url = ngrok.connect(8000)")
    print("  print(f'Public URL: {url}')")
    print("  !python backend/api/server.py")


if __name__ == "__main__":
    setup_colab()