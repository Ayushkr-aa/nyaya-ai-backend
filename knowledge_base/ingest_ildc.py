"""
ILDC Dataset Ingestion Script.
Streams judgments from Hugging Face, chunks them, and embeds into ChromaDB.
"""

import os
import sys
from pathlib import Path
import uuid

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from knowledge_base.chunker import chunk_legal_document, LegalChunk

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

def get_embedding_function():
    if GEMINI_API_KEY:
        print("💡 Using Google Generative AI Embeddings")
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL
        )
    else:
        print("⚠️ GEMINI_API_KEY not found. Using local SentenceTransformers (mismatch possible).")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ingest_ildc(sample_size=100):
    """Download ILDC samples and index them."""
    print(f"📥 Loading ILDC dataset (sample_size={sample_size})...")
    
    # Load 'ILDC' from the 'LilaS/ILDC' repository (usually not gated)
    try:
        dataset = load_dataset("LilaS/ILDC", split=f"test[:{sample_size}]")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Initialize Chroma
    persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    
    # Get or create collection
    # Note: We append to the existing 'indian_laws' collection
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="indian_laws",
        embedding_function=ef
    )

    total_chunks = 0
    print(f"🚀 Starting ingestion of {len(dataset)} cases...")

    for i, entry in enumerate(tqdm(dataset)):
        text = entry.get("text", "")
        case_id = f"ildc_test_{i}"
        
        if not text:
            continue

        # Detect a title for the case if possible
        # ILDC usually has the judgment text starting with case details
        title = text.split("\n")[0][:100].strip() or f"Supreme Court Judgment {i}"
        
        # Chunk the document
        chunks = chunk_legal_document(text)
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []

        for j, chunk in enumerate(chunks):
            batch_ids.append(f"{case_id}_chunk_{j}")
            batch_texts.append(chunk.text)
            
            # Enrich metadata
            meta = chunk.metadata.copy()
            meta.update({
                "source": f"ILDC: {title}",
                "dataset": "ILDC",
                "entry_index": i
            })
            batch_metadatas.append(meta)

        # Upsert in small batches
        if batch_texts:
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            total_chunks += len(batch_texts)

    print(f"\n✅ Finished! Ingested {len(dataset)} cases into {total_chunks} chunks.")
    print(f"📦 ChromaDB location: {persist_path}")

if __name__ == "__main__":
    # You can increase sample_size if needed
    ingest_ildc(sample_size=50)
