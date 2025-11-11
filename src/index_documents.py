from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from uuid import uuid4
from typing import List, Dict, Callable
def upsert_chunks(client:QdrantClient, collection:str, embed:Callable[[str],list], chunks:List[Dict], batch=256):
    buf=[]
    for ch in chunks:
        vec=embed(ch['text']); payload=ch['metadata']; payload.setdefault('text', ch['text'])
        buf.append(PointStruct(id=str(uuid4()), vector=vec, payload=payload))
        if len(buf)>=batch: client.upsert(collection_name=collection, points=buf); buf.clear()
    if buf: client.upsert(collection_name=collection, points=buf)