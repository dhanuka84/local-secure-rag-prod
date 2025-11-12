
## **Overview**

This guide documents how to **build, test, and debug** the full **Local Secure RAG** stack with:

* **LangChain \+ Ollama (Dockerized)**

* **Qdrant (vector DB)**

* **Redis (semantic cache)**

* **Cerbos (role-based auth)**

* **Guardrails / PII filtering**

It includes **real-world debugging steps**, **diagnostic methods**, and **fixes** for cache isolation, typos, and retrieval mismatch.

---

## **🧭 Diagram Overview: Local Secure RAG System**

### **🎯 Purpose:**

Show how:

1. A user query moves through the system

2. Cache, retrieval, and authorization interact

3. Guardrails and PII filters wrap the LLM

4. Role \+ tenant context isolate access

   ---

   ### **🧩 Components to show in diagram**

   #### **🧑‍💻 User**

* Provides query (e.g., *“what are the salary band adjustments?”*)

* Has context: TENANT`=demo`, `APP_ROLE=employee|manager`, `APP_PROFILE=base|guardrails`

  #### **⚙️ Application Layer (`app.py`)**

* Normalizes query (`normalize_query`)

* Redacts PII (`redact_pii`)

* Checks cache (`SemanticCache.get(tenant, role)`)

  #### **💾 Semantic Cache (Redis)**

* Stores `{ query_hash, embedding, answer, sources }`

* Namespaced per tenant \+ role  
   → e.g. `cache:query:demo:manager:123456789`

  #### **🔍 Retriever (Hybrid: Qdrant \+ BM25)**

* Embeddings via `OllamaEmbeddings(nomic-embed-text)`

* Combines dense \+ lexical matches

* Applies `build_prefilter()`

  * Employee → `sensitivity=public`

  * Manager → `sensitivity in [public, confidential]`

  #### **🧑‍⚖️ Cerbos (Policy Engine)**

* Validates user can `read` given documents

* Filters confidential ones for employees

  #### **🧠 LLM (Ollama: `llama3.2`)**

* Consumes contextual snippets

* Generates final answer

  #### **🧱 Authorization & Policy Layer (Cerbos)**

* Enforces document-level access control (ABAC/RBAC) via policies.

* Filters retrieved chunks to only those the current user is allowed to read.

* Runs **pre-LLM** (so restricted content never reaches the model).

* Acts as our primary “guardrail.”

  #### **🪣 Cache Store**

* Final answer stored with tenant-role key

* TTL ensures automatic expiry

  ---

  ### **🔁 Data Flow**

![][image2]

`User Query ─▶ Normalize/PII Clean ─▶ Cache Check (tenant+role)`

   `└── Cache hit? ✅ return answer`

   `└── Cache miss ❌ ▶ Qdrant + BM25 Retrieve`

        `├─▶ Prefilter by tenant+role`

        `├─▶ Cerbos Post-filter`

        `└─▶ Context Construction`

             `▼`

         `LLM (base or guardrails)`

             `▼`

        `Output Guard + PII Redact`

             `▼`

        `Cache.set(tenant, role)`

             `▼`

           `Response`

## **⚙️ 1\. Environment Setup**

### **🧱 Prerequisites**

| Component | Purpose |
| :---- | :---- |
| docker-compose | Run Redis, Qdrant, Ollama, Cerbos |
| Python 3.10+ | Application logic |
| `make` | Helper for pulling Ollama models |
| 8GB+ RAM | Required for LLM embeddings |

---

### **🐳 1️⃣ Start the core stack**

From your project root:

* `Clone the github repo`  
* `cd local-secure-rag-prod-fat`  
* `docker-compose up -d`  
  ![][image3]

This starts:

* **Ollama** (for LLMs and embeddings)

* **Qdrant** (vector DB)

* **Redis** (semantic cache)

* **Cerbos** (authorization service, optional)

  ---

  ### **🧠 2️⃣ Pull required Ollama models**

* `make models`


This command pulls and registers the following inside the Ollama container:

* `llama3.2`

* `nomic-embed-text`

* `llama-guard3` *(for guardrails)*

![][image4]

---

### **🐍 3️⃣ Create & activate Python environment**

* `python -m venv .venv && source .venv/bin/activate`  
* `pip install --upgrade pip`  
* `pip install -r requirements.txt`

`![][image5]`

![][image6]

*(The `requirements.txt` already includes LangChain, Redis, Qdrant, and optional Nemo Guardrails.)*

---

### **🧩 4️⃣ Build your vector index**

* `python src/reindex.py`


This indexes the demo dataset (`sample.txt`, `bm25_corpus.json`) into Qdrant.

Expected output:

* `Reindex complete ✅`  
    
  ---

  ## **🚀 2\. Running the Application**

  ### **Base mode (no guardrails)**

* `source .venv/bin/activate`  
* `python src/app.py`


Output example:

* `--- Local Secure RAG (profile=base, role=employee, tenant=demo) ---`  
* `Ask a question (or 'exit'):`  
    
  ---

  ### **Manager role**

* `APP_PROFILE=guardrails APP_ROLE=manager python src/app.py`


Expected log:

* `✅ NeMo Guardrails: ENABLED`  
* `--- Local Secure RAG (profile=guardrails, role=manager, tenant=demo) ---`  
* `Ask a question (or 'exit'):`

`$ APP_ROLE=manager APP_PROFILE=guardrails  python src/app.py`  
  `NeMo Guardrails init failed (No module named 'nemoguardrails'); using base LLM.`

`--- Local Secure RAG (profile=guardrails, role=manager, tenant=demo) ---`  
`Ask a question (or 'exit'): what are the salary bands adjustments?`

`[DEBUG] === New query ===`  
`[DEBUG] Role=manager, Tenant=demo, Profile=guardrails`  
`[DEBUG] Question: what are the salary bands adjustments?`  
`[DEBUG] Prefilter: should=None min_should=None must=[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo_bounding_box=None, geo_radius=None, geo_polygon=None, values_count=None, is_empty=None, is_null=None)] must_not=None`  
`[DEBUG] Retrieved 3 documents.`  
   `-> 2fa2136d-12fc-49ff-aba9-63e40072c6c6 (sensitivity=confidential)`  
   `-> sample.txt (sensitivity=confidential)`  
   `-> eee92d16-306e-408f-841f-ea97a467a802 (sensitivity=confidential)`  
`[DEBUG] Cerbos not reachable: 'CerbosClient' object has no attribute 'server_info'`  
`[DEBUG] Cerbos allowed_ids: {'2fa2136d-12fc-49ff-aba9-63e40072c6c6', 'eee92d16-306e-408f-841f-ea97a467a802', 'sample.txt'}`  
`[DEBUG] Allowed docs after Cerbos: 3`  
`[DEBUG] Context length: 715 characters`  
`[DEBUG] === End query ===`

`Answer:`  
 `There is no mention of salary band adjustments in the provided CONTEXT. The only information related to adjustments is that <DATE_TIME> salary band adjustments are mentioned as being "internal only".`

`Sources: ['sample.txt', 'sample.txt', 'sample.txt']`  
`Ask a question (or 'exit'): what are the salary bands adjustments?`

`[DEBUG] === New query ===`  
`[DEBUG] Role=manager, Tenant=demo, Profile=guardrails`  
`[DEBUG] Question: what are the salary bands adjustments?`  
`[DEBUG] Cache hit for tenant=demo, role=manager`

`Answer:`  
 `There is no mention of salary band adjustments in the provided CONTEXT. The only information related to adjustments is that <DATE_TIME> salary band adjustments are mentioned as being "internal only".`

`Sources: ['sample.txt', 'sample.txt', 'sample.txt']`  
`Ask a question (or 'exit'): exit`

### **Employee role**

`$ APP_ROLE=employee APP_PROFILE=guardrails  python src/app.py`  
  `NeMo Guardrails init failed (No module named 'nemoguardrails'); using base LLM.`

`--- Local Secure RAG (profile=guardrails, role=employee, tenant=demo) ---`  
`Ask a question (or 'exit'): what are the salary bands adjustments?`

`[DEBUG] === New query ===`  
`[DEBUG] Role=employee, Tenant=demo, Profile=guardrails`  
`[DEBUG] Question: what are the salary bands adjustments?`  
`[DEBUG] Prefilter: should=None min_should=None must=[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo_bounding_box=None, geo_radius=None, geo_polygon=None, values_count=None, is_empty=None, is_null=None), FieldCondition(key='sensitivity', match=MatchValue(value='public'), range=None, geo_bounding_box=None, geo_radius=None, geo_polygon=None, values_count=None, is_empty=None, is_null=None)] must_not=None`  
`[DEBUG] Retrieved 1 documents.`  
   `-> sample.txt (sensitivity=confidential)`  
`[DEBUG] Cerbos not reachable: 'CerbosClient' object has no attribute 'server_info'`  
`[DEBUG] Cerbos allowed_ids: {'sample.txt'}`  
`[DEBUG] Allowed docs after Cerbos: 1`  
`[DEBUG] Context length: 237 characters`  
`[DEBUG] === End query ===`

`Answer:`  
 `According to Document 1, the salary band adjustments are "internal only" and are confidential.`

`Sources: ['sample.txt']`  
`Ask a question (or 'exit'): what are the salary bands adjustments?`

`[DEBUG] === New query ===`  
`[DEBUG] Role=employee, Tenant=demo, Profile=guardrails`  
`[DEBUG] Question: what are the salary bands adjustments?`  
`[DEBUG] Cache hit for tenant=demo, role=employee`

`Answer:`  
 `According to Document 1, the salary band adjustments are "internal only" and are confidential.`

`Sources: ['sample.txt']`  
`Ask a question (or 'exit'): exit`

---

## **🧪 3\. Verification & Testing**

Here’s how to test and verify your entire stack.

### **✅ Check Qdrant**

* `curl http://localhost:6333/collections`


Expected:

* `{"collections":[{"name":"legal_docs"}]}`


  ### **✅ Check Redis (semantic cache)**

* `docker exec -it redis redis-cli`  
* `keys cache:query:*`


You should see:

* `cache:query:demo:manager:573847143880`  
* `cache:query:demo:employee:214486636338`


If you want to reset:

* `flushall`  
* `exit`  
    
  ---

  ### **✅ Ask test questions**

**Employee mode (restricted):**

* `APP_ROLE=employee python src/app.py`  
* `> what are the salary band adjustments?`


Expected:

“`According to Document 1, the salary band adjustments are "internal only" and are confidential`.”

**Manager mode (full access):**

* `APP_ROLE=manager python src/app.py`  
* `> what are the salary band adjustments?`


Expected:

“`There is no mention of salary band adjustments in the provided CONTEXT. The only information related to adjustments is that <DATE_TIME> salary band adjustments are mentioned as being "internal only"`.”

---

### **🧩 Debugging Guardrails and Context**

Enabled debug tracing:

* `print(f"[DEBUG] Role={APP_ROLE}, Tenant={TENANT}, Profile={APP_PROFILE}")`  
* `print(f"[DEBUG] Prefilter: {pf}")`  
* `print(f"[DEBUG] Retrieved {len(results)} documents.")`


When using guardrails:

* `✅ NeMo Guardrails: ENABLED`


If Nemo fails, it gracefully falls back to the base LLM.

---

## **🧰 5\. Redis Debug Commands**

| Command | Description |
| :---- | :---- |
| `keys cache:query:*` | list all cache keys |
| `ttl <key>` | time-to-live for a key |
| `hgetall <key>` | show cached data |
| `flushall` | clear cache |
| `info memory` | check memory usage |

---

## **🧩 6\. Lessons Learned**

1. **Role-based cache isolation is crucial.**

2. **Typos must be normalized before embeddings.**

3. **Guardrails fallback avoids runtime breakage.**

4. **Simple Redis TTL prevents stale cache issues.**

5. **Verbose debugging saves time — always print query, context, and retrieved doc count.**  
   ---

   ## **7\. Combine dense \+ lexical matches**

   

“Combine dense \+ lexical matches” \= hybrid retrieval: run a vector search (dense embeddings) and a keyword/BM25 search (lexical), then fuse the results (e.g., with Reciprocal Rank Fusion), and optionally rerank.

Below is a compact, production-ready way to do this with your stack (Ollama embeddings \+ Qdrant \+ BM25 file).

Please change to **hybrid-retrieval** branch in the repo : [https://github.com/dhanuka84/local-secure-rag-prod/tree/hybrid-retrieval](https://github.com/dhanuka84/local-secure-rag-prod/tree/hybrid-retrieval)

Notes:

* **Dense**: Qdrant vector search via your Ollama embeddings (`nomic-embed-text`).

* **Lexical**: BM25 over a simple tokenized corpus (`bm25_corpus.json`).

* **Fusion**: **RRF** is simple, fast, and robust across domains.

* We keep it dependency-light; add cross-encoder reranking later if needed.

### Quick sanity test

1. Reindex & rebuild corpus:

   $ curl \-X DELETE http://localhost:6333/collections/legal\_docs  
   {"result":true,"status":"ok","time":0.046289722}

`python src/reindex.py`

`python src/build_bm25.py`

2. Run the app (base):

`APP_PROFILE=base APP_ROLE=manager python src/app.py`

3. Ask something with both keywords and semantics:

`“what are the salary band adjustments?”`

`“how are pay bands updated quarterly?”`

You should see slightly **higher recall** and **better robustness to phrasing** than dense-only.

$ APP\_PROFILE=demo APP\_ROLE=manager python src/app.py

\--- Local Secure RAG (profile=demo, role=manager, tenant=demo) \---  
Ask a question (or 'exit'): how are pay bands updated quarterly?

\[DEBUG\] \=== New query \===  
\[DEBUG\] Role=manager, Tenant=demo, Profile=demo  
\[DEBUG\] Question: how are pay bands updated quarterly?  
\[DEBUG\] Prefilter: should=None min\_should=None must=\[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None)\] must\_not=None  
\[DEBUG\] Retrieved 1 documents.  
   \-\> 6ded4f8d-4bb6-47a9-9be4-36c021c90582 (sensitivity=confidential)  
\[DEBUG\] Cerbos not reachable: 'CerbosClient' object has no attribute 'server\_info'  
\[DEBUG\] Cerbos allowed\_ids: {'6ded4f8d-4bb6-47a9-9be4-36c021c90582'}  
\[DEBUG\] Allowed docs after Cerbos: 1  
\[DEBUG\] Context length: 237 characters  
\[DEBUG\] \=== End query \===

Answer:  
 Unfortunately, there is no information in the provided context about updating pay bands. However, it does mention that the \<DATE\_TIME\> salary band adjustments are internal only, implying that they may be handled internally by the organization, but there is no specific information on how they are updated or when updates occur.

Sources: \['sample.txt'\]  
Ask a question (or 'exit'): what are the salary band adjustments?

\[DEBUG\] \=== New query \===  
\[DEBUG\] Role=manager, Tenant=demo, Profile=demo  
\[DEBUG\] Question: what are the salary band adjustments?  
\[DEBUG\] Prefilter: should=None min\_should=None must=\[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None)\] must\_not=None  
\[DEBUG\] Retrieved 1 documents.  
   \-\> 6ded4f8d-4bb6-47a9-9be4-36c021c90582 (sensitivity=confidential)  
\[DEBUG\] Cerbos not reachable: 'CerbosClient' object has no attribute 'server\_info'  
\[DEBUG\] Cerbos allowed\_ids: {'6ded4f8d-4bb6-47a9-9be4-36c021c90582'}  
\[DEBUG\] Allowed docs after Cerbos: 1  
\[DEBUG\] Context length: 237 characters  
\[DEBUG\] \=== End query \===

Answer:  
 The salary band adjustments are described in the following confidential sentence:

"The \<DATE\_TIME\> salary band adjustments are internal only."

Sources: \['sample.txt'\]  
Ask a question (or 'exit'): exit  
(.venv) dhanuka84@dhanuka84:\~/research/local-secure-rag-prod-fat$ APP\_PROFILE=demo APP\_ROLE=employee python src/app.py

\--- Local Secure RAG (profile=demo, role=employee, tenant=demo) \---  
Ask a question (or 'exit'): what are the salary band adjustments?

\[DEBUG\] \=== New query \===  
\[DEBUG\] Role=employee, Tenant=demo, Profile=demo  
\[DEBUG\] Question: what are the salary band adjustments?  
\[DEBUG\] Prefilter: should=None min\_should=None must=\[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None), FieldCondition(key='sensitivity', match=MatchValue(value='public'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None)\] must\_not=None  
\[DEBUG\] Retrieved 1 documents.  
   \-\> 6ded4f8d-4bb6-47a9-9be4-36c021c90582 (sensitivity=None)  
\[DEBUG\] Cerbos not reachable: 'CerbosClient' object has no attribute 'server\_info'  
\[DEBUG\] Cerbos allowed\_ids: {'6ded4f8d-4bb6-47a9-9be4-36c021c90582'}  
\[DEBUG\] Allowed docs after Cerbos: 1  
\[DEBUG\] Context length: 60 characters  
\[DEBUG\] \=== End query \===

Answer:  
 I don't have any information to provide on "salary band adjustments" as it is not present in the provided CONTEXT.

Sources: \['6ded4f8d-4bb6-47a9-9be4-36c021c90582'\]  
Ask a question (or 'exit'): how are pay bands updated quarterly?

\[DEBUG\] \=== New query \===  
\[DEBUG\] Role=employee, Tenant=demo, Profile=demo  
\[DEBUG\] Question: how are pay bands updated quarterly?  
\[DEBUG\] Prefilter: should=None min\_should=None must=\[FieldCondition(key='tenant', match=MatchValue(value='demo'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None), FieldCondition(key='sensitivity', match=MatchValue(value='public'), range=None, geo\_bounding\_box=None, geo\_radius=None, geo\_polygon=None, values\_count=None, is\_empty=None, is\_null=None)\] must\_not=None  
\[DEBUG\] Retrieved 1 documents.  
   \-\> 6ded4f8d-4bb6-47a9-9be4-36c021c90582 (sensitivity=None)  
\[DEBUG\] Cerbos not reachable: 'CerbosClient' object has no attribute 'server\_info'  
\[DEBUG\] Cerbos allowed\_ids: {'6ded4f8d-4bb6-47a9-9be4-36c021c90582'}  
\[DEBUG\] Allowed docs after Cerbos: 1  
\[DEBUG\] Context length: 60 characters  
\[DEBUG\] \=== End query \===

Answer:  
 I don't have direct access to the DOCUMENT 1\. However, I can suggest that according to standard HR practices, pay bands are typically updated on a regular basis, such as during \<DATE\_TIME\> performance reviews or as part of a larger salary review process. The exact timing may vary depending on the organization's policies and procedures.

If you need more specific information on how pay bands are updated in this particular context, I recommend checking DOCUMENT 1 directly for details.

Sources: \['6ded4f8d-4bb6-47a9-9be4-36c021c90582'\]  
Ask a question (or 'exit'): 

---

## **🧱 8\. End-to-End Summary**

| Stage | Command | Description |
| :---- | :---- | :---- |
| **Start stack** | `docker-compose up -d` | Bring up Qdrant, Redis, Ollama |
| **Pull models** | `make models` | Load LLM \+ embedding models |
| **Setup Python** | `python -m venv .venv && source .venv/bin/activate pip install -r requirements.txt` | Environment ready |
| **Index data** | `python src/reindex.py` | Build Qdrant index |
| **Delete index** |  curl \-X DELETE http://localhost:6333/collections/legal\_docs | Delete indexed doc |
| **Run (base)** | `python src/app.py` | Employee role |
| **Run (guardrails)** | `APP_PROFILE=guardrails APP_ROLE=manager python src/app.py` | Manager role with NeMo |
| **Debug cache** | `docker exec -it redis redis-cli` | Inspect cache keys |