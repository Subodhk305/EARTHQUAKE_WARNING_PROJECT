# test_direct_load.py
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

print("=" * 60)
print("Testing direct model loading")
print("=" * 60)

# Import settings
from earthquake_service.config import settings
print(f"\nModel directory from config: {settings.MODEL_DIR}")
print(f"Absolute path: {Path(settings.MODEL_DIR).absolute()}")
print(f"Directory exists: {Path(settings.MODEL_DIR).exists()}")

# Check files
model_path = Path(settings.MODEL_DIR)
if model_path.exists():
    print("\nFiles found:")
    for f in model_path.glob("*"):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
else:
    print(f"\n❌ Directory does not exist: {model_path.absolute()}")

# Try to load models
print("\n" + "=" * 60)
print("Loading models...")
print("=" * 60)

import asyncio
from earthquake_service.services.model_loader import ModelLoader

async def load():
    await ModelLoader.load_all()
    print("\n" + "=" * 60)
    print("Model status:")
    print("=" * 60)
    print(f"Is ready: {ModelLoader.is_ready()}")
    print(f"Info: {ModelLoader.get_model_info()}")

asyncio.run(load())