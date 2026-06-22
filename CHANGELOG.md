# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-21

### Added
- XGBoost demand model with Optuna tuning, time-series cross-validation, and
  optional MLflow tracking.
- Revenue-optimizing pricing engine with a support-aware price search.
- Business-rules layer: hard floors/ceilings, margin floor, max daily swing,
  occupancy-driven directional pressure, manual overrides, and rounding.
- LLM agents (room matching, competitor cleaning, data-quality grading,
  explanation generation) with deterministic fallbacks, plus a LangGraph
  competitor cleaning + quality pipeline.
- Selector-driven scrapers (Playwright, Crawl4AI) with rate limiting,
  robots.txt checks, retries, and incremental refresh.
- Prefect-backed ETL (extract/transform/load) with a parquet feature store and
  PostgreSQL loading.
- FastAPI service: API-key auth, request-id correlation, uniform error handling,
  Prometheus metrics, and background jobs for scrape/ETL/training.
- SQLAlchemy 2.0 typed ORM models and lazy session management.
- CLIs for sample-data generation, training, prediction, and ETL.
- Comprehensive pytest suite, Ruff/Black/isort/mypy configuration, Docker and
  Docker Compose, Grafana/Prometheus provisioning, and GitHub Actions CI.

[1.0.0]: https://github.com/your-org/dynamic-pricing-system/releases/tag/v1.0.0
