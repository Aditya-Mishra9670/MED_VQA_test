import platform
import sys

def check_environment():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed.")
        
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("OpenCV not installed.")

if __name__ == "__main__":
    check_environment()
