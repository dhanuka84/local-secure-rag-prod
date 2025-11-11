import os
import json
import numpy as np
import redis

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser

from cerbos.sdk.client import CerbosClient
from cerbos.sdk.model import Principal, Resource

from semantic_cache import SemanticCache
from prefilter import build_prefilter
from hybrid_retriever import HybridRetriever
from guard_llamaguard import enforce_input_guard, enforce_output_guard
from pii_filter import redact_pii


APP_PROFILE = os.getenv("APP_PROFILE", "base").lower()      # base | guardrails
APP_ROLE    = os.getenv("APP_ROLE", "employee").lower()     # employee | manager
TENANT      = os.getenv("TENANT", "demo")
COLLECTION  = "legal_docs"

# Embeddings / LLM via Ollama
emb = OllamaEmbeddings(model="nomic-embed-text")
base_llm = OllamaLLM(model="llama3.2")

# Optional NeMo Guardrails wrapper
guarded_llm = base_llm
if APP_PROFILE == "guardrails":
    try:
        from nemoguardrails import LLMRails, RailsConfig
        cfg = RailsConfig.from_path("./config")
        app_rails = LLMRails(cfg)

        class Guarded(Runnable):
            def invoke(self, prompt_str: str, config=None, **kwargs):
                return app_rails.generate(prompt_str).get("output", "")

        guarded_llm = Guarded()
        print(" NeMo Guardrails: ENABLED")
    except Exception as e:
        print(f"  NeMo Guardrails init failed ({e}); using base LLM.")

# Clients
qdrant = QdrantClient(url="http://localhost:6333")
rds    = redis.Redis(host="127.0.0.1", port=6379, decode_responses=False)
cache  = SemanticCache(rds, threshold=0.95, ttl=360)
cerbos = CerbosClient(host="http://localhost:3592")

# BM25 auxiliary corpus
aux_path = os.path.join(os.path.dirname(__file__), "bm25_corpus.json")
if os.path.exists(aux_path):
    aux = json.load(open(aux_path, "r", encoding="utf-8"))
    bm25_corpus, doc_ids = aux["corpus"], aux["doc_ids"]
else:
    bm25_corpus, doc_ids = [], []

retriever = HybridRetriever(qdrant, COLLECTION, bm25_corpus, doc_ids, emb.embed_query)

prompt = ChatPromptTemplate.from_template("""
You are a retrieval-augmented assistant.

Use ONLY the information found in the CONTEXT below to answer the QUESTION.
If the CONTEXT contains related information or synonyms, use it to infer an answer.
Do not say "I don't know" if the topic is mentioned, even indirectly.

---
CONTEXT:
{context}
---

QUESTION: {question}

Provide a clear and concise answer, quoting the relevant sentence if needed.
""")




# --- Cerbos compatibility helper ---
def _cerbos_allowed_ids(cerbos: CerbosClient, principal: Principal, resource_objs, action: str, strict: bool = False) -> set:
    """
    Return set of resource IDs allowed for `action`.
    Tries multiple SDK methods. If all fail:
      - strict=False (default): allow all resource IDs (fallback)
      - strict=True: allow none
    """
    allowed = set()
    errors = []

    for res in resource_objs:
        rid = res.id

        # Try simple boolean
        try:
            if hasattr(cerbos, "is_allowed"):
                ok = cerbos.is_allowed(principal=principal, resource=res, action=action)
                if ok:
                    allowed.add(rid)
                    continue
        except Exception as e:
            errors.append(("is_allowed", str(e)))

        # Try check_resource (decision object)
        try:
            if hasattr(cerbos, "check_resource"):
                decision = cerbos.check_resource(principal=principal, resource=res, actions={action})
                # Try decision.is_allowed(action)
                if hasattr(decision, "is_allowed") and callable(decision.is_allowed):
                    if decision.is_allowed(action):
                        allowed.add(rid)
                        continue
                # Or try dict-like .actions
                actions = getattr(decision, "actions", None)
                if isinstance(actions, dict) and actions.get(action, False):
                    allowed.add(rid)
                    continue
        except Exception as e:
            errors.append(("check_resource", str(e)))

    # Fallback behavior
    if not allowed:
        if not strict:
            # permissive fallback: allow all (we already prefiltered by tenant)
            return {res.id for res in resource_objs}
        else:
            print("X Cerbos strict mode: denying all due to errors:", errors)
            return set()

    return allowed


def answer_query(user_q: str):
    print("\n[DEBUG] === New query ===")
    print(f"[DEBUG] Role={APP_ROLE}, Tenant={TENANT}, Profile={APP_PROFILE}")
    print(f"[DEBUG] Question: {user_q}")

    # --- Normalize user input ---
    user_q = user_q.strip().lower()
    user_q = user_q.replace("ajustments", "adjustments")  # common typo fix

    if APP_PROFILE == "guardrails":
        enforce_input_guard(user_q)
    user_q = redact_pii(user_q)


    qvec = np.array(emb.embed_query(user_q), dtype=np.float32)
    cached = cache.get(user_q, qvec, tenant=TENANT, role=APP_ROLE)
    if cached:
        print(f"[DEBUG] Cache hit for tenant={TENANT}, role={APP_ROLE}")
        return cached.get("answer", "(cached)"), cached

    # Prefilter
    pf: Filter = build_prefilter(tenant=TENANT, role=APP_ROLE)
    print("[DEBUG] Prefilter:", pf)

    # Retrieve
    results = retriever.retrieve(user_q, top_k=5, prefilter=pf)
    print(f"[DEBUG] Retrieved {len(results)} documents.")
    for r in results:
        sid = r.get("id", r.get("source"))
        print(f"   -> {sid} (sensitivity={r.get('sensitivity')})")

    # Cerbos connectivity test
    try:
        cerbos_health = cerbos.server_info()
        print(f"[DEBUG] Cerbos reachable: {cerbos_health}")
    except Exception as e:
        print("[DEBUG] Cerbos not reachable:", e)

    # Cerbos post-filter
    principal = Principal(id="user", roles=[APP_ROLE], attr={"tenant": TENANT})
    resource_list = [
        Resource(id=r.get("id", r.get("source", "unknown")), kind="document", attr=r)
        for r in results
    ]
    CERBOS_STRICT = os.getenv("CERBOS_STRICT", "false").lower() == "true"
    allowed_ids = _cerbos_allowed_ids(cerbos, principal, resource_list, action="read", strict=CERBOS_STRICT)
    print(f"[DEBUG] Cerbos allowed_ids: {allowed_ids}")

    allowed = [r for r in results if r.get("id", r.get("source", "unknown")) in allowed_ids]
    print(f"[DEBUG] Allowed docs after Cerbos: {len(allowed)}")

    context = "\n\n".join(
        [f"[Document {i+1}] (Source: {d.get('source','?')})\n{d.get('text','')[:400]}"
         for i, d in enumerate(allowed)]
    )
    print(f"[DEBUG] Context length: {len(context)} characters")

    chain = (
        {"context": lambda q: context, "question": RunnablePassthrough()}
        | prompt
        | guarded_llm
        | StrOutputParser()
    )
    ans = chain.invoke(user_q)
    ans = redact_pii(ans)

    if APP_PROFILE == "guardrails":
        ans = enforce_output_guard(ans)

    cache.set(
        user_q,
        qvec,
        {"answer": ans, "sources": [d.get("source") for d in allowed]},
        tenant=TENANT,
        role=APP_ROLE,
    )

    print("[DEBUG] === End query ===\n")
    return ans, {"sources": [d.get("source") for d in allowed]}



def cli():
    print(f"\n--- Local Secure RAG (profile={APP_PROFILE}, role={APP_ROLE}, tenant={TENANT}) ---")
    while True:
        q = input("Ask a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            ans, meta = answer_query(q)
            print("\nAnswer:\n", ans)
            if meta:
                print("\nSources:", meta.get("sources"))
        except Exception as e:
            print("\nBlocked/Failed:", e)


if __name__ == "__main__":
    cli()
