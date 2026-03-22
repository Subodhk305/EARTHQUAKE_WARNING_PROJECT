# check_cuda.py
import torch
import sys

print("=" * 60)
print("🔍 CUDA Availability Check")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("❌ CUDA not available - training will be slower on CPU")
    
print("=" * 60)

# Also check for cuDNN
if torch.backends.cudnn.is_available():
    print("✅ cuDNN available for accelerated training")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")