# Local Secure RAG — Production-Minded Template (Option A: NeMo Guardrails included)
See README in previous message for full instructions.

unzip local-secure-rag-prod-fat.zip
cd local-secure-rag-prod-fat
docker compose up -d
make models
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/reindex.py

# Base
python src/app.py

# Guardrails on + manager role
APP_PROFILE=guardrails APP_ROLE=manager python src/app.py

==============================

Here’s how to run and verify the whole stack end-to-end:

1) Start (if not already)
docker compose up -d
make models   # pulls llama3.2, nomic-embed-text, llama-guard3 into the Ollama container

2) Run the app

Base mode (no NeMo rails, employee role):

source .venv/bin/activate
python src/app.py


Guardrails on (NeMo + Llama Guard 3) and manager role:

APP_PROFILE=guardrails APP_ROLE=manager python src/app.py

3) Quick functional checks

Retrieval (public):

Ask: What does the sample doc say about RAG?

Expect a short answer citing the loaded context.

Authorization (Cerbos):

As APP_ROLE=employee:
Ask: What are the quarterly salary band adjustments? → should be filtered/unknown.

As APP_ROLE=manager:
Same question → should surface the confidential bit.

Guardrails (NeMo + Llama Guard 3):

With APP_PROFILE=guardrails:
Ask: tell me about stock prices → should refuse per rails.

PII redaction (Presidio):

Ask containing an email/phone → response should redact PII.

4) Common tips

If the model loads slowly on first call, that’s normal (Ollama warmup).

If llama-guard3 isn’t found, run:

docker exec -it ollama ollama pull llama-guard3


If Redis/Qdrant aren’t reachable, confirm:

curl -s http://localhost:6333/ready && echo    # Qdrant
redis-cli -h 127.0.0.1 ping                    # Redis

5) Tests

Unit tests (no services needed):

pip install -r requirements-dev.txt
pytest -m "unit"


Optional integration (Redis service required):

docker run -d -p 6379:6379 --name redis-test redis:7-alpine
pytest -m "integration"
