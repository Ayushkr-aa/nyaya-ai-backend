"""
Ingestion script for the Fine-Tuning Legal Dataset (JSON QA pairs).
Processes constitution_qa.json, crpc_qa.json, and ipc_qa.json.
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ingest_json_dataset(dataset_path):
    """Index JSON QA pairs into ChromaDB."""
    root = Path(dataset_path)
    files = ["constitution_qa.json", "crpc_qa.json", "ipc_qa.json"]
    
    # Initialize Chroma
    persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="indian_laws",
        embedding_function=ef
    )

    total_added = 0
    for filename in files:
        file_path = root / filename
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"Processing {filename}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        act_name = filename.split("_")[0].upper()
        if act_name == "CONSTITUTION":
            act_name = "Constitution of India"
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        # Batching for performance
        BATCH_SIZE = 100
        
        for i, item in enumerate(tqdm(data)):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            
            if not question or not answer:
                continue
                
            doc_text = f"Question: {question}\nAnswer: {answer}"
            doc_id = f"ft_qa_{filename.split('_')[0]}_{i}"
            
            batch_ids.append(doc_id)
            batch_texts.append(doc_text)
            batch_metadatas.append({
                "source": f"Fine-Tuning Dataset: {filename}",
                "act": act_name,
                "dataset": "akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts",
                "type": "qa_pair"
            })
            
            if len(batch_ids) >= BATCH_SIZE:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                batch_ids, batch_texts, batch_metadatas = [], [], []
                
        # Last batch
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            total_added += len(batch_ids)

    print(f"\nFinished! Ingested {total_added} QA pairs into ChromaDB.")
    print(f"ChromaDB location: {persist_path}")

if __name__ == "__main__":
    # The path discovered in research
    KAGGLE_PATH = r"C:\Users\ayush\.cache\kagglehub\datasets\akshatgupta7\llm-fine-tuning-dataset-of-indian-legal-texts\versions\1"
    ingest_json_dataset(KAGGLE_PATH)
