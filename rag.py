
import os
import pickle
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MIN_SCORE = 0.2

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(question, index, chunks, model, k=TOP_K):
    query_vec = model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= MIN_SCORE:
            results.append({"text": chunks[idx]["text"], "source": chunks[idx]["source"], "score": float(score)})
    return results

def build_prompt(question, retrieved):
    if not retrieved:
        return None
    context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in retrieved)
    return f"""You are an academic assistant for IIT Bombay students.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly: "I don't know based on the available documents."
Do not make up any information.

Context:
{context}

Question: {question}
Answer:"""

def answer(question, index, chunks, embed_model, llm_client):
    retrieved = retrieve(question, index, chunks, embed_model)
    prompt = build_prompt(question, retrieved)
    if prompt is None:
        return "I don't know based on the available documents.", []
    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip(), retrieved

def init():
    llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    embed_model = SentenceTransformer(EMBED_MODEL)
    index, chunks = load_index()
    return embed_model, llm_client, index, chunks