"""
Kaggle Dataset Ingestion Script.
Downloads Supreme Court judgment PDFs from Kaggle, extracts text, 
chunks them, and embeds into ChromaDB.
"""

import os
import sys
from pathlib import Path
from pypdf import PdfReader
import kagglehub
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions
from knowledge_base.chunker import chunk_legal_document

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

def get_embedding_function():
    print("💡 Using local SentenceTransformers for embeddings (stable).")
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extract and clean text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path.name}: {e}")
        return ""

def ingest_kaggle_data(sample_size=30):
    """Download Kaggle dataset and index PDF content."""
    print("📥 Downloading latest version of SC Judgments dataset...")
    try:
        # Using the specific dataset provided by the user
        dataset_path = kagglehub.dataset_download("adarshsingh0903/legal-dataset-sc-judgments-india-19502024")
        print(f"✅ Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return

    dataset_root = Path(dataset_path)
    # The dataset contains many PDFs, likely in subdirectories by year
    pdf_files = list(dataset_root.glob("**/*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDF documents.")

    if not pdf_files:
        print("⚠️ No PDF files found in the dataset.")
        return

    # Process a sample to keep it fast for documentation/demo
    selected_pdfs = pdf_files[:sample_size]
    print(f"🚀 Processing first {len(selected_pdfs)} documents...")

    # Initialize Chroma
    persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="indian_laws",
        embedding_function=ef
    )

    total_chunks = 0
    for pdf_path in tqdm(selected_pdfs):
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            continue

        filename = pdf_path.name
        case_id = f"sc_judge_{pdf_path.stem}"
        
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
                "source": f"Supreme Court Judgment: {filename}",
                "dataset": "Kaggle: SC Judgments (1950-2024)",
                "filename": filename
            })
            batch_metadatas.append(meta)

        # Ingest
        if batch_texts:
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            total_chunks += len(batch_texts)

    print(f"\n🎉 Finished! Ingested {len(selected_pdfs)} PDFs into {total_chunks} chunks.")
    print(f"📦 ChromaDB location: {persist_path}")

if __name__ == "__main__":
    ingest_kaggle_data(sample_size=20) # Start with 20 for a quick test
