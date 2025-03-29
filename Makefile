# Makefile for RAG-HW project

.PHONY: run analyze clean setup

run:
	python main.py

analyze:
	python analytics.py

setup:
	pip install -r requirements.txt

clean:
	rm -f query_log.json
	rm -rf Analytics audio transcripts faiss_index
