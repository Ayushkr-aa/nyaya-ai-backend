"""
Local LLM inference module.
Uses llama-cpp-python to run GGUF models on CPU.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_llm_instance = None


def get_local_llm():
    """Get or create the local LLM instance (lazy loading)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        )

    model_path = os.getenv("LOCAL_MODEL_PATH", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    full_path = Path(__file__).resolve().parent.parent / model_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Local model not found at: {full_path}\n"
            "Please run download_model.py to fetch the model."
        )

    print(f"🤖 Loading local model: {full_path.name} ...")
    _llm_instance = Llama(
        model_path=str(full_path),
        n_ctx=2048,          # Context window
        n_threads=4,         # CPU threads (adjust to your CPU)
        n_gpu_layers=0,      # 0 = CPU only (no GPU)
        verbose=False,
    )
    print(f"✅ Local model loaded! ({full_path.stat().st_size / 1024**2:.0f} MB)")
    return _llm_instance


def generate_local(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate a response using the local GGUF model."""
    llm = get_local_llm()

    # Format as Qwen ChatML template
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.7,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    response = output["choices"][0]["text"].strip()
    return response


def is_local_model_available() -> bool:
    """Check if the local model file exists."""
    model_path = os.getenv("LOCAL_MODEL_PATH", "./models/doj-legal-gemma-1b.gguf")
    full_path = Path(__file__).resolve().parent.parent / model_path
    return full_path.exists()
