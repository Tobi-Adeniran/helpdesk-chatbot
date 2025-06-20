from flask import Flask, request, jsonify
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
from llama_cpp import Llama

# Load FAISS index + chunk data
index = faiss.read_index("helpdesk.index")
with open("helpdesk.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
meta   = data["meta"]

# Load the embedding model on CPU (Intel GPU won't work with CUDA)
enb_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Initialize LLaMA2 with 8-bit quantized GGUF on CPU and increased context
llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_gpu_layers=0,  # CPU only
    n_threads=8,
    n_ctx=2048       # Increased context window (512 is default)
)

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        # Embed query
        query_embedding = enb_model.encode([query])

        # Retrieve top-1 chunk for smaller prompt size
        D, I = index.search(query_embedding, k=1)
        context = chunks[I[0][0]][:1000]  # Truncate to 1,000 characters

        # Prompt format
        prompt = f"""You are a helpful assistant trained on helpdesk logs. Use the context below to answer the user's question.

Context:
{context}

User question:
{query}

Answer:"""

        # Run model inference
        resp = llm(prompt=prompt, max_tokens=256)
        answer = resp["choices"][0]["text"].strip()

    except Exception as e:
        print(f"Error during inference: {e}")
        answer = "Sorry, I couldn't generate a response at the moment."

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
