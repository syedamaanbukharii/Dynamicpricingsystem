# AI-Powered Dynamic Hotel Pricing System

A production-grade platform that recommends **revenue-maximizing nightly room
prices**. It combines an **XGBoost** demand model with a deterministic,
rule-aware pricing engine, and uses Large Language Models **only** for
unstructured-data cleaning, normalization, and natural-language explanations
**never** for the numeric price decision.

> Clone it, generate sample data, train a model, and get explainable price
> recommendations from a REST API in a few minutes. No external services are
> required to run locally — optional dependencies degrade gracefully.

---

## Table of contents

- [Why this design](#why-this-design)
- [Features](#features)
- [Architecture](#architecture)
- [Quickstart (local)](#quickstart-local)
- [Quickstart (Docker)](#quickstart-docker)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Development](#development)
- [Testing](#testing)
- [Observability](#observability)
- [A note on scraping](#a-note-on-scraping)
- [License](#license)

## Why this design

- **XGBoost predicts demand, not the LLM.** The model predicts `rooms_sold`
  (units of demand) for a feature vector that includes the candidate price. The
  engine searches prices, computes `expected_revenue = price × predicted demand`
  (capped at inventory), picks the revenue-optimal price, then applies business
  rules. Pricing stays deterministic, fast, and auditable.
- **No train/serve skew.** A single deterministic `FeatureBuilder` produces
  features for both training and serving, and both the trainer and predictor
  assert the model's feature names match the canonical `FEATURE_ORDER`.
- **Runs out of the box.** With no trained model it serves a heuristic demand
  curve; with no parquet engine it writes CSV; with no LLM key it produces
  rule-based explanations; with no Prefect it runs the ETL in-process; with no
  database it skips persistence with a warning.

## Features

- XGBoost demand model with **Optuna** hyperparameter tuning and time-series
  cross-validation; optional **MLflow** tracking.
- Revenue-optimizing **pricing engine** with a support-aware price search.
- **Business-rules layer**: hard floors/ceilings, margin floor, max daily price
  swing, occupancy-driven directional pressure, manual overrides, and rounding.
- **LLM agents** (Claude): room-name matching, competitor-rate cleaning,
  data-quality grading, and explanation generation — all with deterministic
  fallbacks. A **LangGraph** pipeline orchestrates competitor cleaning + quality.
- **Selector-driven scrapers** (Playwright for JS sites, Crawl4AI for static),
  with rate limiting, robots.txt checks, retries, and incremental refresh.
- **Prefect-backed ETL** (extract → transform → load) writing a parquet feature
  store and PostgreSQL.
- **FastAPI** service with API-key auth, request-id correlation, uniform error
  responses, Prometheus metrics, and background jobs.
- **SQLAlchemy 2.0** typed ORM (parameterized queries by construction).
- Full **typing + docstrings**, **Ruff/Black/isort/mypy**, **pytest** suite,
  **Docker**/**Docker Compose**, and a **GitHub Actions** CI pipeline.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full write-up and
[`docs/diagrams.md`](docs/diagrams.md) for component, sequence, ETL, and training
diagrams. In short:

```
api → services → {pricing, training, scraping, etl, inference, agents}
                       → {feature_engineering, schemas, utils, config}
```

## Quickstart (local)

Requires **Python 3.12**.

```bash
# 1) Create a virtualenv and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (optional, for live scraping) install the browser:
python -m playwright install chromium

# 2) Configure (optional — sensible defaults work for local)
cp .env.example .env

# 3) Generate sample data, train a model, and make a prediction
python -m scripts.generate_sample_data
python -m scripts.train_model
python -m scripts.run_prediction --hotel HOTEL_A --room DELUXE_KING \
    --inventory 50 --sold 20 --previous-price 180 --competitors 175,189,205,210

# 4) Run the API
uvicorn app.api.main:app --reload
# open http://localhost:8000/docs
```

Even before training, the API works using the heuristic demand model:

```bash
curl -s -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"hotel_id":"HOTEL_A","room_type":"DELUXE_KING","stay_date":"2026-07-15",
       "inventory_total":50,"rooms_sold":20,"previous_price":180,
       "competitor_rates":[175,189,205,210]}' | jq
```

## Quickstart (Docker)

Brings up the API, PostgreSQL, Redis, Prometheus, Grafana, and MLflow:

```bash
docker compose up -d --build
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
# MLflow:     http://localhost:5000
```

Generate data and train inside the running container:

```bash
docker compose exec app python -m scripts.generate_sample_data
docker compose exec app python -m scripts.train_model
```

## Configuration

All settings come from environment variables (a `.env` file is read in local
development). Secrets use `SecretStr` and are never logged. The full list with
defaults is in [`.env.example`](.env.example); highlights:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `local` | `local`/`development`/`staging`/`production`/`test` |
| `API_KEY` | `change-me-in-production` | API key; auth bypassed for the default key when not in production |
| `ANTHROPIC_API_KEY` | _(empty)_ | Enables LLM cleaning/explanations; empty ⇒ rule-based |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model used by the agents |
| `LLM_ENABLED` | `true` | Master switch for LLM use |
| `POSTGRES_*` | `pricing` | Database connection |
| `OPTUNA_TRIALS` | `40` | Hyperparameter search budget (set `0` to skip tuning) |
| `MODEL_DIR` | `./data/models` | Where trained models are persisted |

## CLI reference

| Command | Description |
| --- | --- |
| `python -m scripts.generate_sample_data [--days N] [--out DIR]` | Write a sample raw-data drop to `data/raw` |
| `python -m scripts.train_model [--data PATH] [--val-fraction F] [--no-db]` | Train and persist the XGBoost model |
| `python -m scripts.run_prediction [...]` | Produce a single recommendation (JSON) |
| `python -m scripts.run_etl [--no-db] [--default-stay DATE]` | Run the ETL pipeline |

These are also exposed as console scripts after `pip install -e .`:
`pricing-generate-data`, `pricing-train`, `pricing-predict`, `pricing-etl`.

## API reference

Prefix: `/api/v1`. Full examples in [`docs/api_examples.md`](docs/api_examples.md).

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/health` | no | Liveness/readiness + model availability |
| POST | `/recommendations` | yes | Recommend a revenue-optimal price |
| POST | `/explanations` | yes | Explain a recommended price |
| POST | `/scrape` | yes | Trigger a competitor scrape (background) |
| POST | `/etl/run` | yes | Trigger the ETL pipeline (background) |
| POST | `/training/run` | yes | Trigger model (re)training (background) |
| GET | `/metrics` | no | Prometheus exposition (served at root) |

Interactive docs: `/docs` (Swagger) and `/redoc`.

## Project layout

```
dynamic-pricing-system/
├── app/
│   ├── api/                 FastAPI app, middleware, deps, routers
│   ├── agents/              LLM agents + LangGraph competitor pipeline
│   ├── config/              Pydantic-settings configuration
│   ├── database/            SQLAlchemy ORM models + lazy sessions
│   ├── etl/                 extract / transform / load + Prefect flow
│   ├── feature_engineering/ deterministic FeatureBuilder + calendars
│   ├── inference/           demand-model protocol, XGBoost predictor, heuristic
│   ├── monitoring/          Prometheus metrics (no-op fallback)
│   ├── pricing/             business rules + revenue-optimizing engine
│   ├── schemas/             Pydantic domain + API contracts
│   ├── scraping/            selector-driven scrapers + rate limiting
│   ├── services/            application-facing orchestration
│   ├── training/            dataset prep, Optuna tuning, trainer, metrics
│   └── utils/               logging, exceptions, retry
├── data/{raw,processed,features,models}/
├── docker/                  entrypoint, prometheus, grafana provisioning
├── docs/                    architecture, diagrams, API examples
├── scripts/                 CLIs + sample-data generator
├── tests/                   pytest suite
├── .github/workflows/ci.yml CI: lint, type-check, test on Python 3.12
├── Dockerfile  docker-compose.yml  Makefile
├── pyproject.toml  requirements.txt  requirements-dev.txt
└── .env.example  .gitignore  LICENSE  CONTRIBUTING.md  CHANGELOG.md  SECURITY.md
```

## Development

```bash
pip install -r requirements-dev.txt && pip install -e .
make format     # isort + black + ruff --fix
make lint       # ruff
make typecheck  # mypy
make test       # pytest
make check      # lint + typecheck + test
```

## Testing

```bash
make cov   # pytest with coverage (term + HTML in htmlcov/)
```

The suite covers business rules, the pricing engine, feature engineering and
`FEATURE_ORDER` integrity, the agents (offline), scraping (fake backend), the ETL
transform, the API surface (including heuristic-fallback prediction and auth),
utilities, schema validators, and the real XGBoost train→persist→load→predict
path.

## Observability

- **Logs**: Loguru, optional JSON, per-request `X-Request-ID`.
- **Metrics**: Prometheus at `/metrics`; a starter Grafana dashboard is
  provisioned in `docker/grafana`.
- **Decision auditability**: each recommendation includes applied constraints and
  feature drivers.

## A note on scraping

The repository ships **example** scrape targets and CSS selectors in
[`app/scraping/targets.yaml`](app/scraping/targets.yaml) with placeholder URLs.
Scraping is intentionally **configuration-driven**: there are no hardcoded
site-specific selectors in the code. Before enabling any target, review the
destination site's Terms of Service and `robots.txt`, set realistic rate limits,
and confirm you are permitted to collect the data. `SCRAPE_RESPECT_ROBOTS` is on
by default.

## License

Released under the [MIT License](LICENSE).
# Dynamicpricingsystem
