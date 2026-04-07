"""
Document ingestion script.
Reads legal text files, chunks them, and embeds into ChromaDB.
"""

import os
import sys
from pathlib import Path

# Add parent dir to path so we can import from knowledge_base
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from knowledge_base.chunker import chunk_legal_document

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"


# Local embeddings via sentence-transformers
from chromadb.utils import embedding_functions
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_chroma_client():
    """Get or create a persistent ChromaDB client."""
    persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    return client


def ingest_all():
    """Read all .txt files from documents/, chunk them, embed, and store in ChromaDB."""
    client = get_chroma_client()

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection("indian_laws")
        print("🗑️  Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="indian_laws",
        metadata={"description": "Indian legal documents for RAG"},
        embedding_function=local_ef
    )

    total_chunks = 0

    for txt_file in sorted(DOCUMENTS_DIR.glob("*.txt")):
        print(f"\n📄 Processing: {txt_file.name}")
        content = txt_file.read_text(encoding="utf-8")
        chunks = chunk_legal_document(content)
        print(f"   → {len(chunks)} chunks extracted")

        # Batch embed and insert
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [f"{txt_file.stem}_{total_chunks + j}" for j in range(len(batch))]
            texts = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            total_chunks += len(batch)
            print(f"   ✅ Ingested batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\n🎉 Done! Total chunks ingested: {total_chunks}")
    print(f"📦 ChromaDB persisted at: {CHROMA_PERSIST_DIR}")
    return total_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("  DoJ Legal Knowledge Base — Ingestion Script")
    print("=" * 60)
    ingest_all()
