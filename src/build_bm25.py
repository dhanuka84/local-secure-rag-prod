import json, os
from qdrant_client import QdrantClient

OUT = os.path.join(os.path.dirname(__file__), "bm25_corpus.json")
COLLECTION = "legal_docs"

def main():
    qc = QdrantClient(url="http://localhost:6333")

    # Pull all points’ payloads (or paginate if you have many)
    points, _ = qc.scroll(collection_name=COLLECTION, limit=100000, with_payload=True)

    corpus, doc_ids = [], []
    for p in points:
        text = (p.payload or {}).get("text", "")
        if not text: 
            continue
        corpus.append(text)
        doc_ids.append(str(p.id) if p.id is not None else (p.payload.get("source","unknown")))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "doc_ids": doc_ids}, f, ensure_ascii=False)

    print(f"BM25 corpus built: {len(corpus)} docs -> {OUT}")

if __name__ == "__main__":
    main()
