import os
import pickle
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = "data"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
OVERLAP = 80

def load_documents(data_dir):
    docs = []
    for filepath in Path(data_dir).iterdir():
        if filepath.suffix == ".txt":
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "source": filepath.name})
        elif filepath.suffix == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            docs.append({"text": text, "source": filepath.name})
    print(f"Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    step = chunk_size - overlap
    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        for i in range(0, len(text), step):
            piece = text[i:i + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "source": source})
            if i + chunk_size >= len(text):
                break
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors.")
    return index, embeddings

def save(index, chunks):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index to {INDEX_FILE} and chunks to {CHUNKS_FILE}.")

if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    index, embeddings = build_index(chunks, model)
    save(index, chunks)
    print("Ingestion complete!")