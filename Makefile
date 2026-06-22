# Makefile for the AI-Powered Dynamic Hotel Pricing System.
# Usage: `make <target>`. Run `make help` to list targets.

.DEFAULT_GOAL := help
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: help install install-dev playwright generate-data train predict etl \
        run lint format typecheck test cov check docker-build docker-up \
        docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install dev + runtime dependencies (editable)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

playwright: ## Install Playwright browser binaries (for live scraping)
	$(PYTHON) -m playwright install chromium

generate-data: ## Generate sample raw data into data/raw
	$(PYTHON) -m scripts.generate_sample_data

train: ## Train the demand model from data/raw/observations.csv
	$(PYTHON) -m scripts.train_model

predict: ## Produce a sample price recommendation
	$(PYTHON) -m scripts.run_prediction

etl: ## Run the ETL pipeline over data/raw
	$(PYTHON) -m scripts.run_etl

run: ## Run the API with auto-reload (development)
	$(PYTHON) -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

lint: ## Lint with Ruff
	$(PYTHON) -m ruff check app scripts tests

format: ## Auto-format with isort + Black + Ruff --fix
	$(PYTHON) -m isort app scripts tests
	$(PYTHON) -m black app scripts tests
	$(PYTHON) -m ruff check --fix app scripts tests

typecheck: ## Type-check with mypy
	$(PYTHON) -m mypy app

test: ## Run the test suite
	$(PYTHON) -m pytest

cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Run lint + typecheck + tests

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the full stack (app, postgres, redis, prometheus, grafana, mlflow)
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
