import numpy as np
import redis
import time
from typing import Optional, Dict, List


class SemanticCache:
    """
    Tenant- and role-aware semantic cache for query/embedding/results.
    Ensures isolation across tenants and roles.
    """

    def __init__(self, redis_client: redis.Redis, threshold: float = 0.95, ttl: int = 3600):
        self.redis = redis_client
        self.threshold = threshold
        self.ttl = ttl

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _key(self, query: str, tenant: str, role: str) -> str:
        tenant = tenant.lower().strip()
        role = role.lower().strip()
        qhash = abs(hash(query)) % (10 ** 12)
        return f"cache:query:{tenant}:{role}:{qhash}"

    # ------------------------------------------------------------------ #
    # GET
    # ------------------------------------------------------------------ #
    def get(
        self,
        query: str,
        query_embedding: np.ndarray,
        tenant: str,
        role: str
    ) -> Optional[Dict]:
        """
        Retrieve cached data for this (tenant, role, query).
        Returns None if not found or similarity < threshold.
        """
        cache_key = self._key(query, tenant, role)
        cached_data = self.redis.hgetall(cache_key)
        if not cached_data:
            return None

        try:
            cached_embedding = np.frombuffer(cached_data[b'embedding'], dtype=np.float32)
        except Exception:
            return None

        sim = self._cosine_similarity(query_embedding, cached_embedding)
        if sim < self.threshold:
            return None

        return {
            "answer": cached_data.get(b'answer', b'').decode(),
            "results": cached_data.get(b'results', b'').decode(),
            "sources": cached_data.get(b'sources', b'').decode() if b'sources' in cached_data else None,
            "cache_hit": True,
            "similarity": sim
        }

    # ------------------------------------------------------------------ #
    # SET
    # ------------------------------------------------------------------ #
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        data: Dict,
        tenant: str,
        role: str,
        ttl: Optional[int] = None
    ):
        """
        Store cache entry specific to this (tenant, role).
        """
        cache_key = self._key(query, tenant, role)
        ttl = ttl or self.ttl
        mapping = {
            "query": query,
            "embedding": query_embedding.tobytes(),
            "answer": str(data.get("answer", "")),
            "results": str(data.get("results", "")),
            "sources": str(data.get("sources", "")),
            "timestamp": str(time.time()),
        }
        self.redis.hset(cache_key, mapping=mapping)
        self.redis.expire(cache_key, ttl)

    # ------------------------------------------------------------------ #
    # CLEAR (optional utility)
    # ------------------------------------------------------------------ #
    def clear(self, tenant: Optional[str] = None, role: Optional[str] = None) -> int:
        """
        Clear cached entries, scoped by tenant and/or role.
        Returns number of deleted keys.
        """
        if tenant and role:
            pattern = f"cache:query:{tenant}:{role}:*"
        elif tenant:
            pattern = f"cache:query:{tenant}:*"
        elif role:
            pattern = f"cache:query:*:{role}:*"
        else:
            pattern = "cache:query:*"

        keys = self.redis.keys(pattern)
        deleted = 0
        if keys:
            deleted = self.redis.delete(*keys)
        return deleted
