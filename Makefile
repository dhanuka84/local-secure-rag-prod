.PHONY: up down logs models reindex run run-guardrails
up:
	docker compose up -d
down:
	docker compose down
logs:
	docker compose logs -f
models:
	docker exec -it ollama ollama pull llama3.2 || true
	docker exec -it ollama ollama pull nomic-embed-text || true
	docker exec -it ollama ollama pull llama-guard3 || true
reindex:
	python src/reindex.py
run:
	APP_ROLE?=employee TENANT?=demo APP_PROFILE?=base \
	APP_ROLE=$(APP_ROLE) TENANT=$(TENANT) APP_PROFILE=$(APP_PROFILE) python src/app.py
run-guardrails:
	APP_ROLE?=employee TENANT?=demo \
	APP_ROLE=$(APP_ROLE) TENANT=$(TENANT) APP_PROFILE=guardrails python src/app.py
