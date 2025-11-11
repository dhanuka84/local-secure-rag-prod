from qdrant_client.models import Filter, FieldCondition, MatchValue
def build_prefilter(tenant:str, role:str):
    must=[FieldCondition(key='tenant', match=MatchValue(value=tenant))]
    if role=='employee': must.append(FieldCondition(key='sensitivity', match=MatchValue(value='public')))
    return Filter(must=must)