# Architecture

This document explains how the AI-Powered Dynamic Hotel Pricing System is put
together, why the major decisions were made, and how data flows from raw inputs
to a published price recommendation.

## Guiding principles

1. **The price model is XGBoost, not an LLM.** Demand is predicted by a
   gradient-boosted regression model. Large Language Models are used *only* for
   unstructured-data cleaning, normalization, and natural-language explanation —
   never for the numeric price decision. This keeps pricing deterministic,
   auditable, and cheap to evaluate.
2. **Train/serve symmetry.** Features are built by a single deterministic
   `FeatureBuilder`, and the exact ordered feature list (`PricingFeatures.FEATURE_ORDER`)
   is the one source of truth. The trainer and the predictor both assert that the
   model's feature names match this list, so training-serving skew is impossible
   by construction.
3. **Graceful degradation.** Optional dependencies (MLflow, the Anthropic SDK,
   Playwright, Crawl4AI, Prefect, Prometheus client, a parquet engine, a live
   database) are all import-guarded. A freshly cloned checkout runs end-to-end
   with sensible fallbacks: a heuristic demand curve when no model is trained,
   CSV when no parquet engine is present, rule-based explanations when no LLM
   key is configured, and in-process execution when Prefect is absent.
4. **Separation of concerns.** Domain logic (features, model, rules, engine)
   knows nothing about HTTP, persistence, or orchestration. Services compose
   domain logic; the API and CLIs are thin adapters over services.

## The core pricing idea

The demand model predicts **`rooms_sold`** (units of demand) as a function of a
feature vector that *includes the candidate price*. Given a request, the pricing
engine:

1. Builds a price-search band anchored on competitor context (falling back to the
   previous price, then the midpoint of the business band). The band is
   intentionally bounded near observed support because tree models extrapolate
   flat outside their training range — probing arbitrarily high would yield a
   spurious "higher is always better" optimum.
2. Evaluates expected revenue `= price × predicted_rooms_sold` (capped at
   remaining inventory) across a coarse grid, then refines around the best point.
3. Selects the revenue-maximizing price — the **unconstrained optimal**.
4. Passes that price through the **business-rules layer**, which enforces hard
   floors/ceilings, a margin floor, a maximum day-over-day swing, occupancy-driven
   directional pressure, optional manual overrides, and rounding, producing the
   final **recommended price**.

Every decision records which constraints bound it and the top feature drivers, so
the result is fully auditable.

## Layered structure

```
app/
  config/         Pydantic-settings configuration (env-driven)
  utils/          logging (Loguru), typed exceptions, retry/backoff
  schemas/        Pydantic domain models + API request/response contracts
  feature_engineering/  deterministic FeatureBuilder, calendars, defaults
  inference/      DemandModel protocol, XGBoost predictor, heuristic fallback
  pricing/        business rules + revenue-optimizing engine
  training/       dataset prep, Optuna tuning, XGBoost trainer, evaluation
  agents/         LLM client + room-matching, competitor-cleaning,
                  data-quality, feature-engineering, explanation agents,
                  and a LangGraph competitor pipeline
  scraping/       selector-driven scrapers (Playwright / Crawl4AI) + rate limiting
  etl/            extract / transform / load + Prefect-backed flow
  database/       SQLAlchemy 2.0 ORM models + lazy session management
  monitoring/     Prometheus metrics with no-op fallback
  services/       application-facing orchestration (pricing, training, scraping, etl)
  api/            FastAPI app, middleware, dependencies, routers
```

Dependencies flow downward: `api → services → {pricing, training, scraping, etl,
inference, agents} → {feature_engineering, schemas, utils, config}`. Nothing in
the domain layers imports from `api` or `services`.

## Data flow

### Offline (data → model)

1. **Extract** reads raw inputs (observations, competitor listings, events,
   holidays, inventory, bookings) from `data/raw` via a pluggable `DataSource`.
2. **Transform** canonicalizes room types, cleans and structures competitor
   listings (cleaning + quality agents), normalizes timestamps, dedupes, drops
   invalid prices, and builds the engineered feature matrix.
3. **Load** writes the feature matrix to the feature store (`data/features`) and
   cleaned entities to PostgreSQL (best-effort).
4. **Train** prepares a chronological train/validation split, tunes XGBoost
   hyperparameters with Optuna (TimeSeriesSplit cross-validation), fits the
   model, evaluates RMSE/MAE/R²/MAPE, and persists `model.joblib` + metadata,
   updating `latest.txt`. The run is optionally logged to MLflow and recorded in
   the `model_runs` table.

### Online (request → recommendation)

1. A client calls `POST /api/v1/recommendations` with a self-contained request.
2. The pricing service loads the latest model (or the heuristic fallback),
   builds features with the shared builder, runs the engine, applies business
   rules, and — when requested — asks the explanation agent for a business-
   friendly rationale (LLM when configured, deterministic otherwise).
3. The response includes the recommended price, the unconstrained optimum,
   expected occupancy and revenue, the effective floor/ceiling, the applied
   constraints, the feature drivers, the model version, and the explanation.

## Configuration

All configuration is environment-driven through `app/config/settings.py`
(`pydantic-settings`). Nested settings use the `__` delimiter and a `.env` file
is supported for local development. Secrets (`API_KEY`, `POSTGRES_PASSWORD`,
`ANTHROPIC_API_KEY`) are typed as `SecretStr` and never logged. See
`.env.example` for the full list.

## Observability

* **Logging**: Loguru with optional JSON output and a per-request correlation id
  bound via middleware and echoed in the `X-Request-ID` response header.
* **Metrics**: Prometheus counters/histograms/gauges for request volume and
  latency, predictions, scrape/ETL/training runs, and the loaded model version,
  exposed at `/metrics`.
* **Tracing of decisions**: every recommendation carries its applied constraints
  and feature drivers, providing decision-level auditability.

## Testing strategy

The suite covers the business-rules layer (floors, ceilings, swing limits,
overrides, occupancy pressure, rounding), the pricing engine (bounded output,
interior optima, monotonic demand), feature engineering (single-row/matrix parity
and `FEATURE_ORDER` integrity), the agents (offline rule-based paths), scraping
(selector-driven parsing and incremental skipping with a fake backend), the ETL
transform, the API surface (health, prediction via heuristic fallback, validation
errors, metrics, auth), the utilities (retry/backoff, exception serialization),
the schema validators, and the real XGBoost train→persist→load→predict path.
