from flask import Flask, request, jsonify
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
from flask_cors import CORS


# Load FAISS index + chunk data
index = faiss.read_index("helpdesk.index")
with open("helpdesk.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
meta   = data["meta"]

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Embed query
    query_embedding = model.encode([query])
    
    # Search FAISS index
    D, I = index.search(query_embedding, k=3)
    context_chunks = [chunks[i] for i in I[0]]

    # Combine top context
    context = "\n\n".join(context_chunks)

    # Prepare LLaMA prompt
    prompt = f"""You are a helpful assistant trained on helpdesk logs. Use the context below to answer the user's question.

Context:
{context}

User question:
{query}

Answer:"""

    # Send to local Ollama model
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt, "stream": False}
    )
    reply = response.json().get("response", "No response from LLaMA.")

    return jsonify({"answer": reply.strip()})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
