# Makefile for AI Resume Analyser & Job Matching Platform
.PHONY: help setup run test lint eval docker-up clean

PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff
MYPY := mypy

help:
	@echo "Available commands:"
	@echo "  make setup      - Install production and development dependencies"
	@echo "  make run        - Start FastAPI backend and Streamlit UI in parallel"
	@echo "  make test       - Execute full pytest suite with 80% coverage check"
	@echo "  make lint       - Run Ruff linter and Mypy static type checker"
	@echo "  make eval       - Run ML model benchmark evaluation and sync metrics"
	@echo "  make docker-up  - Build and launch production Docker multi-container stack"
	@echo "  make clean      - Remove cache artifacts, coverage files, and bytecode"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) -c "from app.services.vector_indexer import build_or_load_vector_index; build_or_load_vector_index()"

run:
	$(PYTHON) run.py

test:
	$(PYTEST) tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=80

lint:
	$(RUFF) check app frontend tests
	$(MYPY) app --ignore-missing-imports

eval:
	$(PYTHON) evaluate_models.py

docker-up:
	docker-compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true
