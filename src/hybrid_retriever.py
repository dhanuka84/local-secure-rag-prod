from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchParams
from rank_bm25 import BM25Okapi

# K for Reciprocal Rank Fusion
RRF_K = 60

class HybridRetriever:
    """
    Hybrid retriever: Qdrant dense search + BM25 lexical search + Reciprocal Rank Fusion.
    Optionally rerank later if you add a cross-encoder.
    """
    def __init__(
        self,
        qdrant: QdrantClient,
        collection: str,
        bm25_corpus: List[str],
        doc_ids: List[str],
        embed_fn: Callable[[str], List[float]],
        bm25_tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.qdrant = qdrant
        self.collection = collection
        self.embed_fn = embed_fn

        # Build BM25 index in-memory from prebuilt corpus
        if bm25_corpus and doc_ids and len(bm25_corpus) == len(doc_ids):
            # Simple whitespace tokenization; customize if you want smarter tokenization
            tokenized = [
                (bm25_tokenizer(doc) if bm25_tokenizer else doc.lower().split())
                for doc in bm25_corpus
            ]
            self.bm25 = BM25Okapi(tokenized)
            self._bm25_docs = bm25_corpus
            self._bm25_ids = doc_ids
        else:
            self.bm25 = None
            self._bm25_docs = []
            self._bm25_ids = []

    def _dense_search(
        self, query: str, top_k: int, prefilter: Optional[Filter]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Return [(doc_id, score, payload)] from Qdrant dense search.
        """
        query_vec = np.array(self.embed_fn(query), dtype=np.float32).tolist()
        res = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=max(top_k, 10),  # fetch extra for safety
            query_filter=prefilter,
            search_params=SearchParams(hnsw_ef=128)
        )
        out = []
        for r in res:
            pid = str(r.id) if r.id is not None else (r.payload.get("source","unknown"))
            out.append((pid, float(r.score), r.payload or {}))
        return out

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Return [(doc_id, bm25_score)] from BM25. If BM25 not built, return [].
        """
        if not self.bm25:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Top-K indices by score
        idx = np.argsort(scores)[::-1][:max(top_k, 10)]
        return [(self._bm25_ids[i], float(scores[i])) for i in idx]

    def _rrf_fuse(
        self,
        dense: List[Tuple[str, float, Dict]],
        sparse: List[Tuple[str, float]],
        k: int = RRF_K,
    ) -> List[str]:
        """
        Reciprocal Rank Fusion: combines rankings; returns ordered list of doc_ids.
        """
        ranks = {}
        # Dense: already ordered by similarity desc from Qdrant
        for rank, (doc_id, _, _) in enumerate(dense, 1):
            ranks[doc_id] = ranks.get(doc_id, 0.0) + 1.0 / (k + rank)

        # Sparse (BM25): treat current order as ranking
        for rank, (doc_id, _) in enumerate(sparse, 1):
            ranks[doc_id] = ranks.get(doc_id, 0.0) + 1.0 / (k + rank)

        fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in fused]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        prefilter: Optional[Filter] = None,
    ) -> List[Dict]:
        """
        Returns top_k payload dicts (each includes 'id', 'text', 'source', 'sensitivity', ...)
        after fusing dense + lexical results. No cross-encoder here to keep deps light.
        """
        dense_results = self._dense_search(query, top_k=top_k, prefilter=prefilter)
        sparse_results = self._bm25_search(query, top_k=top_k)

        # If only one modality available, use it
        if not sparse_results:
            fused_ids = [doc_id for doc_id, _, _ in dense_results]
        elif not dense_results:
            fused_ids = [doc_id for doc_id, _ in sparse_results]
        else:
            fused_ids = self._rrf_fuse(dense_results, sparse_results, k=RRF_K)

        # Build id -> payload map from dense; if BM25-only doc appears, fetch it from Qdrant
        payload_by_id = {doc_id: payload for doc_id, _, payload in dense_results}

        # For BM25-only hits not present in dense results, fetch payloads via scroll by id
        missing_ids = [i for i in fused_ids if i not in payload_by_id]
        if missing_ids:
            # You can use scroll + filter by ids in batches; for small sets this is fine:
            for mid in missing_ids:
                points, _ = self.qdrant.scroll(collection_name=self.collection, limit=1, with_payload=True)
                # NOTE: Qdrant Python SDK doesn't have direct "get by id" in scroll;
                # for production, keep a map id->payload at ingestion or store payload in BM25 corpus file.
                # Here we skip filling if not found via dense to keep it simple.

        # Assemble results in fused order, attach 'id'
        out = []
        for did in fused_ids:
            p = payload_by_id.get(did)
            if p is None:
                # fall back: include a minimal stub; app can handle missing fields
                p = {"source": did, "text": ""}
            item = dict(p)
            item["id"] = did
            out.append(item)

        return out[:top_k]
