import os
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────────────────────
LOGS_DIR     = "helpdesk_logs"     # folder with your .txt log files
EMBED_MODEL  = "all-MiniLM-L6-v2"  # compact embedding model
CHUNK_SIZE   = 500                 # approx token limit per chunk
INDEX_PATH   = "helpdesk.index"
PICKLE_PATH  = "helpdesk.pkl"
# ────────────────────────────────────────────────────────────────────────────

def load_txt(path):
    return Path(path).read_text(encoding="utf-8")

def chunk_text(text, max_len=CHUNK_SIZE):
    """Split on blank lines to keep logical entries together."""
    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        # word‑count check
        if len((cur + p).split()) < max_len:
            cur += p + "\n\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = p + "\n\n"
    if cur:
        chunks.append(cur.strip())
    return chunks

def main():
    # 1. Load all helpdesk .txt files
    texts, filenames = [], []
    for file in Path(LOGS_DIR).glob("*.txt"):
        texts.append(load_txt(file))
        filenames.append(file.name)
    print(f"Loaded {len(texts)} helpdesk log files.")

    # 2. Chunk each log
    all_chunks, chunk_meta = [], []
    for text, name in zip(texts, filenames):
        for c in chunk_text(text):
            all_chunks.append(c)
            chunk_meta.append(name)
    print(f"Created {len(all_chunks)} chunks from logs.")

    # 3. Embed chunks
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    # 4. Build & save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    # 5. Persist chunks + metadata
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": chunk_meta}, f)

    print(f"Index saved to {INDEX_PATH}, chunks to {PICKLE_PATH}.")

if __name__ == "__main__":
    main()
