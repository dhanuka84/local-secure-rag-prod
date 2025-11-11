from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, ScalarType

def _build_scalar_quant():
    """
    Build a scalar quantization object compatible with the installed qdrant-client.
    Tries several class/name variants; falls back to None (no quantization).
    """
    # 1) Most versions accept this:
    try:
        from qdrant_client.models import ScalarQuantization
        return ScalarQuantization(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    except Exception:
        pass

    # 2) Some versions want a dict-like payload:
    try:
        return {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}}
    except Exception:
        pass

    # 3) Give up (no quantization)
    return None

def ensure_collection(client: QdrantClient, name: str, dim: int = 768):
    cols = client.get_collections().collections
    if any(c.name == name for c in cols):
        return

    quant_cfg = _build_scalar_quant()

    kwargs = dict(
        collection_name=name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,   # keep base vectors on disk
        ),
    )

    # Only pass quantization_config if we built something valid
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg

    client.create_collection(**kwargs)
