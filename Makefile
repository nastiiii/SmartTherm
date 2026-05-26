PYTHON ?= ./.venv/bin/python

.PHONY: init test demo bot admin pipeline drafts install faq-expand kb-sync kb-build wiki-serve ollama-pull rag-test eval-hallucinations

install:
	$(PYTHON) -m pip install -r requirements.txt

init:
	$(PYTHON) scripts/init_all.py

test:
	$(PYTHON) -m pytest tests/ -q

demo:
	$(PYTHON) scripts/demo_retrieval_batch.py

eval-gold:
	$(PYTHON) scripts/build_eval_gold.py

eval:
	$(PYTHON) scripts/build_eval_gold.py
	$(PYTHON) scripts/eval_retrieval.py

bot:
	$(PYTHON) -m app.bot.main

admin:
	$(PYTHON) -m app.admin.api

pipeline:
	$(PYTHON) scripts/run_pipeline.py

drafts:
	$(PYTHON) scripts/expand_faq_drafts.py
	$(PYTHON) scripts/load_draft_candidates.py

faq-expand:
	$(PYTHON) scripts/build_faq_extended.py
	$(PYTHON) scripts/clean_faq_seed.py
	$(PYTHON) scripts/init_all.py

faq-clean:
	$(PYTHON) scripts/clean_faq_seed.py
	$(PYTHON) scripts/init_all.py

faq-audit: faq-clean
	$(PYTHON) scripts/build_wiki_site.py

kb-build:
	$(PYTHON) scripts/build_faq_extended.py
	$(PYTHON) scripts/clean_faq_seed.py

kb-sync:
	$(PYTHON) scripts/sync_knowledge.py

WIKI_PORT ?= 8091

wiki-serve:
	@echo "Wiki: http://127.0.0.1:$(WIKI_PORT)/  (override: make wiki-serve WIKI_PORT=9000)"
	$(PYTHON) -m http.server $(WIKI_PORT) --directory wiki/site

OLLAMA_MODEL ?= qwen2.5:7b-instruct

ollama-pull:
	docker compose exec ollama ollama pull $(OLLAMA_MODEL)

rag-test:
	$(PYTHON) scripts/test_rag_ollama.py

eval-hallucinations:
	$(PYTHON) scripts/eval_hallucinations.py
