import os
import sys
from pathlib import Path
import kagglehub
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions
from knowledge_base.chunker import chunk_legal_document
import pandas as pd

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

def get_embedding_function():
    print("💡 Using local SentenceTransformers for embeddings (stable).")
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def extract_text_from_file(file_path):
    """Extract text depending on file extension."""
    ext = file_path.suffix.lower()
    text = ""
    try:
        if ext == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif ext == '.txt' or ext == '.csv':
            if ext == '.csv':
                # For CSV, read it and convert contents to string
                df = pd.read_csv(file_path)
                text = df.to_string()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
    return text.strip()

def ingest_constitution():
    """Download Constitution Kaggle dataset and index content."""
    print("📥 Downloading Constitution of India dataset...")
    try:
        dataset_path = kagglehub.dataset_download("rushikeshdarge/constitution-of-india")
        print(f"✅ Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return

    dataset_root = Path(dataset_path)
    
    # Grab any viable documents
    files = []
    files.extend(list(dataset_root.glob("**/*.pdf")))
    files.extend(list(dataset_root.glob("**/*.txt")))
    files.extend(list(dataset_root.glob("**/*.csv")))
    
    print(f"📂 Found {len(files)} document files.")

    if not files:
        print("⚠️ No valid files (.pdf, .txt, .csv) found in the dataset.")
        # Print what files *are* there to debug
        all_files = list(dataset_root.glob("**/*"))
        print("Files present instead:", [f.name for f in all_files if f.is_file()])
        return

    # Initialize Chroma
    persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="indian_laws",
        embedding_function=ef
    )

    total_chunks = 0
    for doc_path in tqdm(files):
        text = extract_text_from_file(doc_path)
        if not text or len(text) < 100:
            continue

        filename = doc_path.name
        doc_id = f"const_{doc_path.stem}"
        
        # Chunk the document using the legal chunker
        chunks = chunk_legal_document(text)
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []

        for j, chunk in enumerate(chunks):
            batch_ids.append(f"{doc_id}_chunk_{j}")
            batch_texts.append(chunk.text)
            
            # Enrich metadata
            meta = chunk.metadata.copy()
            meta.update({
                "source": f"Constitution Document: {filename}",
                "dataset": "Kaggle: Constitution of India",
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

    print(f"\n🎉 Finished! Ingested {len(files)} files into {total_chunks} Constitution chunks.")
    print(f"📦 ChromaDB updated at: {persist_path}")

if __name__ == "__main__":
    ingest_constitution()
