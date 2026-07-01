import torch

def test_gpu():
    if not torch.cuda.is_available():
        print("CUDA is NOT available. Running on CPU is not recommended for this project.")
        return
        
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

if __name__ == "__main__":
    test_gpu()
