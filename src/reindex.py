import os, json
from qdrant_client import QdrantClient
from langchain_ollama.embeddings import OllamaEmbeddings
from semantic_chunker import SemanticChunker
from qdrant_setup import ensure_collection
from index_documents import upsert_chunks

COLLECTION = "legal_docs"   # keep same collection name

def embed_fn(text: str):
    return OllamaEmbeddings(model="nomic-embed-text").embed_query(text)

def main():
    qdrant = QdrantClient(url="http://localhost:6333")
    ensure_collection(qdrant, COLLECTION, dim=768)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus, doc_ids = [], []
    chunker = SemanticChunker()

    for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(data_dir, fname)
        text = open(path, "r", encoding="utf-8").read()

        # Base metadata (without sensitivity)
        base_meta = {
            "source": fname,
            "doc_type": "contract",
            "year": 2023,
            "tenant": os.getenv("TENANT", "demo"),
        }

        chunks = chunker.chunk(text, base_meta)

        # Set sensitivity PER CHUNK + store text in payload
        for ch in chunks:
            t = ch["text"]
            ch["metadata"]["sensitivity"] = (
                "confidential" if "[confidential]" in t.lower() else "public"
            )
            ch["metadata"]["text"] = t  # used by retriever/reranker

        upsert_chunks(qdrant, COLLECTION, embed_fn, chunks, batch=128)
        corpus.append(text)
        doc_ids.append(fname)

    # Save BM25 aux
    with open(os.path.join(os.path.dirname(__file__), "bm25_corpus.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_ids": doc_ids, "corpus": corpus}, f)

    print("Reindex complete ✅")

if __name__ == "__main__":
    main()
