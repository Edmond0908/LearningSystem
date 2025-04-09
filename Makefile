.PHONY: run analyze clean setup lint format

run:
	python main.py

analyze:
	python analytics.py

setup:
	pip install -r requirements.txt

clean:
	rm -f query_log.json
	rm -rf Analytics audio transcripts faiss_index

lint:
	pre-commit run --all-files

format:
	black . && isort . && autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
