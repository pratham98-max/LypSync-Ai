import torch
import time
from TTS.api import TTS

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Simple matrix multiplication to spike GPU
        print("Running GPU spike test...")
        a = torch.randn(10000, 10000).cuda()
        b = torch.randn(10000, 10000).cuda()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"GPU matrix mult took: {time.time() - start:.4f}s")
        
    else:
        print("CUDA NOT AVAILABLE")

if __name__ == "__main__":
    check_gpu()
