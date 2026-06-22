# API examples

Base URL (local): `http://localhost:8000`
API prefix: `/api/v1`

Authentication: side-effectful and pricing endpoints expect an `X-API-Key`
header. For local/non-production runs where `API_KEY` is left at its default, the
check is bypassed for convenience. In production a correct key is always
required.

## Health

```bash
curl -s http://localhost:8000/api/v1/health | jq
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "local",
  "model_available": false,
  "timestamp": "2026-06-21T12:00:00+00:00"
}
```

`model_available` is `false` when no trained model is present and the heuristic
fallback is serving; it becomes `true` after a successful training run.

## Recommend a price

```bash
curl -s -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "hotel_id": "HOTEL_A",
    "room_type": "DELUXE_KING",
    "stay_date": "2026-07-15",
    "inventory_total": 50,
    "rooms_sold": 20,
    "previous_price": 180.0,
    "competitor_rates": [175, 189, 205, 210],
    "booking_velocity": 3.0,
    "include_explanation": true
  }' | jq
```

```json
{
  "hotel_id": "HOTEL_A",
  "room_type": "DELUXE_KING",
  "stay_date": "2026-07-15",
  "currency": "USD",
  "recommended_price": 198.0,
  "unconstrained_optimal_price": 246.31,
  "expected_occupancy": 0.62,
  "expected_revenue": 6138.0,
  "effective_floor": 144.0,
  "effective_ceiling": 216.0,
  "price_change_pct": 0.10,
  "manual_override_applied": false,
  "applied_constraints": [
    {"rule": "max_daily_change_pct", "description": "Up-move limited to 20% of previous price (180.00)."}
  ],
  "feature_drivers": {"price": 0.31, "demand_score": 0.30, "competitor_median": 0.09},
  "model_version": "model_20260621_120000",
  "explanation": {
    "summary": "Recommended 198.00 USD for DELUXE_KING ...",
    "drivers": ["Competitor median is 197.00.", "Up-move limited to 20% of previous price."],
    "generated_by": "rule-based"
  }
}
```

(Numbers are illustrative; exact values depend on the trained model and inputs.)

### Field notes

* `competitor_rates` is an optional list of already-cleaned nightly rates. Omit it
  if you have none; the engine falls back to the previous price / business band.
* `business_rules` may be supplied inline to override defaults per request, e.g.
  `"business_rules": {"min_rate": 120, "max_rate": 320, "manual_override": 250}`.
* Setting `manual_override` forces that price (clamped only by hard bounds).

## Explain a price

```bash
curl -s -X POST http://localhost:8000/api/v1/explanations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{ "hotel_id": "HOTEL_A", "room_type": "DELUXE_KING",
        "stay_date": "2026-07-15", "inventory_total": 50, "rooms_sold": 20,
        "previous_price": 180.0 }' | jq
```

## Trigger a competitor scrape (background)

```bash
curl -s -X POST http://localhost:8000/api/v1/scrape \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{ "stay_dates": ["2026-07-15", "2026-07-16"], "incremental": true }' | jq
```

```json
{ "job_id": "5f0c...", "job_type": "scrape", "status": "accepted",
  "detail": "Scraping 2 stay date(s).", "submitted_at": "2026-06-21T12:00:00+00:00" }
```

## Trigger the ETL pipeline (background)

```bash
curl -s -X POST http://localhost:8000/api/v1/etl/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{ "persist_to_db": true }' | jq
```

## Trigger model training (background)

```bash
curl -s -X POST http://localhost:8000/api/v1/training/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{ "data_path": "data/raw/observations.csv", "val_fraction": 0.2, "reload_after": true }' | jq
```

When `reload_after` is `true`, the serving model is hot-reloaded once training
finishes, so subsequent recommendations use the new model without a restart.

## Metrics

```bash
curl -s http://localhost:8000/metrics
```

Returns Prometheus text exposition (request counts/latency, predictions,
scrape/ETL/training run counters, loaded model version gauge).

## Interactive docs

FastAPI serves OpenAPI docs at `http://localhost:8000/docs` (Swagger UI) and
`http://localhost:8000/redoc`.
