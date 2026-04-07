"""Test script for training data generation."""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

from training.generate_dataset import SEED_QA_PAIRS, OUTPUT_FILE, generate_qa_from_chunk
from knowledge_base.chunker import chunk_legal_document

# Save seed pairs first
with open(str(OUTPUT_FILE), "w", encoding="utf-8") as f:
    for pair in SEED_QA_PAIRS:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
print(f"✅ Saved {len(SEED_QA_PAIRS)} seed pairs")

# Try generating from first IPC chunk
doc_path = Path("knowledge_base/documents/ipc.txt")
content = doc_path.read_text(encoding="utf-8")
chunks = chunk_legal_document(content)
print(f"IPC has {len(chunks)} chunks")

# Try generating from a specific chunk
for chunk in chunks[:3]:
    if len(chunk.text) < 100:
        continue
    print(f"\nGenerating for: {chunk.metadata.get('section', 'Unknown')}...")
    pairs = generate_qa_from_chunk(chunk.text, chunk.metadata, n=2)
    if pairs:
        print(f"  ✅ Got {len(pairs)} pairs")
        for p in pairs:
            print(f"  Q: {p.get('instruction', 'N/A')[:60]}")
        # Append to file
        with open(str(OUTPUT_FILE), "a", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    else:
        print("  ⚠️ No pairs (rate limit or parse error)")
    
    import time
    time.sleep(3)

# Count total
with open(str(OUTPUT_FILE), "r", encoding="utf-8") as f:
    total = sum(1 for _ in f)
print(f"\n📊 Total pairs in file: {total}")
