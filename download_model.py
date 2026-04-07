"""
Script to instantly download a tiny footprint offline model.
Downloads the Gemma 3 1B Instruct model in Q4_K_M GGUF format (approx 750 MB).
"""
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    import subprocess
    import sys
    print("Installing huggingface-hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download


def download_model():
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    dest_path = models_dir / filename
    if dest_path.exists():
        print(f"✅ Model already exists at: {dest_path}")
        return str(dest_path)
    
    print(f"⬇️ Downloading offline AI model ({filename})...")
    print("   This is about 750 MB, give it a few minutes depending on your internet.")
    
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False
    )
    
    print(f"🎉 Download complete! Model saved to: {downloaded_path}")
    return downloaded_path


if __name__ == "__main__":
    download_model()
