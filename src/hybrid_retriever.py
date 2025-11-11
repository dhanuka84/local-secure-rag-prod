from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever:
    def __init__(self, qdrant: QdrantClient, coll: str, bm25_corpus: List[str], doc_ids: List[str], embed_fn):
        self.qdrant = qdrant
        self.collection = coll
        self.embed = embed_fn
        self.bm25 = BM25Okapi([t.split() for t in bm25_corpus]) if bm25_corpus else None
        self.doc_ids = doc_ids
        self.cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, top_k=10, prefilter: Filter=None) -> List[Dict]:
        # Dense
        qvec = self.embed(query)
        dense = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k * 3,
            query_filter=prefilter,
        )

        # Sparse (BM25 -> filenames)
        sparse_ids = []
        if self.bm25 and self.doc_ids:
            scores = self.bm25.get_scores(query.split())
            top_idx = np.argsort(scores)[::-1][:top_k * 3]
            sparse_ids = [self.doc_ids[i] for i in top_idx]  # filenames like 'sample.txt'

        # Reciprocal Rank Fusion on a unified key set
        fused = self._rrf([d.id for d in dense], sparse_ids, k=60)

        # Pull texts (by id if UUID; else by source filter)
        texts = [self._get_text(doc_key) for doc_key in fused[:20]]
        pairs = [[query, t] for t in texts]
        ce_scores = self.cross.predict(pairs)

        ranked = sorted(
            zip(fused[:20], ce_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [self._to_doc(doc_key, score) for doc_key, score in ranked]

    def _rrf(self, dense_ids: List[str], sparse_ids: List[str], k=60) -> List[str]:
        rank = {}
        for r, d in enumerate(dense_ids, 1):
            rank[d] = rank.get(d, 0) + 1 / (k + r)
        for r, s in enumerate(sparse_ids, 1):
            rank[s] = rank.get(s, 0) + 1 / (k + r)
        return [d for d, _ in sorted(rank.items(), key=lambda x: x[1], reverse=True)]

    # --- helpers that tolerate either UUID point IDs OR source filenames ---

    def _get_text(self, key: str) -> str:
        # try by point id
        try:
            pts = self.qdrant.retrieve(self.collection, ids=[key])
            if pts and pts[0] and pts[0].payload:
                return pts[0].payload.get("text", "")
        except Exception:
            pass
        # fallback: search by source == key
        payload = self._get_first_payload_by_source(key)
        return payload.get("text", "") if payload else ""

    def _to_doc(self, key: str, score: float) -> Dict:
        # try by point id
        try:
            pts = self.qdrant.retrieve(self.collection, ids=[key])
            if pts and pts[0]:
                payload = pts[0].payload or {}
                payload["id"] = str(pts[0].id)
                payload["score"] = float(score)
                return payload
        except Exception:
            pass
        # fallback by source
        payload = self._get_first_payload_by_source(key) or {}
        payload.setdefault("id", key)  # keep the filename key if no UUID
        payload["score"] = float(score)
        return payload

    def _get_first_payload_by_source(self, source_value: str) -> Dict:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_value))])
        # scroll returns (points, next_page_offset)
        try:
            points, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return points[0].payload or {}
        except Exception:
            pass
        return {}
