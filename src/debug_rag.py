from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from prefilter import build_prefilter
from hybrid_retriever import HybridRetriever
from langchain_ollama.embeddings import OllamaEmbeddings
import json, os

COLLECTION = "legal_docs"
TENANT = os.getenv("TENANT", "demo")
APP_ROLE = os.getenv("APP_ROLE", "manager")

emb = OllamaEmbeddings(model="nomic-embed-text")
qdrant = QdrantClient(url="http://localhost:6333")

# 1️ Check collection info
print("Collections:")
print(qdrant.get_collections())

# 2️ Prefilter build
pf = build_prefilter(tenant=TENANT, role=APP_ROLE)
print("\nPrefilter built:", pf)

# 3️ Inspect first few points
print("\nSample points (showing text & sensitivity):")
points, _ = qdrant.scroll(collection_name=COLLECTION, limit=3, with_payload=True)
for p in points:
    print(f"- id: {p.id}")
    print(f"  sensitivity: {p.payload.get('sensitivity')}")
    print(f"  source: {p.payload.get('source')}")
    print(f"  text preview: {p.payload.get('text', '')[:120]}")

# 4️ Try manual search
retriever = HybridRetriever(qdrant, COLLECTION, [], [], emb.embed_query)
query = "salary band adjustments"
print("\nRetrieving for query:", query)
results = retriever.retrieve(query, top_k=3, prefilter=pf)
print(f"Results found: {len(results)}")
for r in results:
    print(f"  -> {r.get('source')} | sensitivity={r.get('sensitivity')} | text preview={r.get('text','')[:100]}")
