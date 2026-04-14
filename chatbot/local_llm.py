"""
Local LLM inference module.
Uses Hugging Face transformers and kagglehub to run native local models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_llm_model = None
_llm_tokenizer = None


def get_local_llm():
    """Get or create the local LLM pipeline (lazy loading)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None and _llm_tokenizer is not None:
        return _llm_model, _llm_tokenizer

    try:
        import kagglehub
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(f"Missing required ML libraries. Error: {e}")

    # Use the model requested by the user
    model_id = "google/gemma-4/transformers/gemma-4-26b-a4b"
    
    print(f"📥 Downloading or verifying model weights from KaggleHub: {model_id} ...")
    print(f"⚠️  WARNING: This is a over 26B parameters model. It may require immense System RAM/VRAM!")
    
    try:
        path = kagglehub.model_download(model_id)
        print("✅ Path to model files:", path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model via kagglehub: {e}")

    print(f"🤖 Loading tokenizer and local model weights into memory...")
    
    _llm_tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Load model with automatic device mapping to utilize GPU if available
    # Using float16 to significantly reduce memory bandwidth overhead.
    _llm_model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16, 
    )
    
    print(f"✅ Local Transformer model loaded successfully!")
    return _llm_model, _llm_tokenizer


def generate_local(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate a response using the Hugging Face Transformers model."""
    model, tokenizer = get_local_llm()
    import torch

    formatted_prompt = f"User: {prompt}\nAssistant: "

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature), # Do sample requires > 0
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated part mapped past the input
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def is_local_model_available() -> bool:
    """Always return True to enable Transformers local fallback lazily."""
    return True
