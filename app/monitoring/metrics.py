"""Prometheus metrics.

Defines the application's metric instruments and a renderer for the ``/metrics``
endpoint. The ``prometheus_client`` dependency is import-guarded: when it is not
installed, fully API-compatible no-op stand-ins are used so instrumentation
calls throughout the codebase never fail and the endpoint still responds (with a
short notice instead of an exposition payload).
"""

from __future__ import annotations

from typing import Any

from app.utils.logging import get_logger

logger = get_logger("api")

try:  # pragma: no cover - exercised only when prometheus_client is installed
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path in minimal envs
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoOpMetric:
        """A no-op metric exposing the subset of the API the app uses."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value = 0.0

        def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
            return self

        def inc(self, amount: float = 1.0) -> None:
            self._value += amount

        def dec(self, amount: float = 1.0) -> None:
            self._value -= amount

        def set(self, value: float) -> None:
            self._value = value

        def observe(self, amount: float) -> None:
            self._value = amount

    class CollectorRegistry:  # type: ignore[no-redef]
        """Placeholder registry used when prometheus_client is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def Counter(*args: Any, **kwargs: Any) -> _NoOpMetric:  # type: ignore[misc]
        """No-op Counter factory used when prometheus_client is absent."""
        return _NoOpMetric()

    def Gauge(*args: Any, **kwargs: Any) -> _NoOpMetric:  # type: ignore[misc]
        """No-op Gauge factory used when prometheus_client is absent."""
        return _NoOpMetric()

    def Histogram(*args: Any, **kwargs: Any) -> _NoOpMetric:  # type: ignore[misc]
        """No-op Histogram factory used when prometheus_client is absent."""
        return _NoOpMetric()

    def generate_latest(registry: Any = None) -> bytes:  # type: ignore[misc]
        """Return a placeholder exposition payload when the client is absent."""
        return b"# prometheus_client not installed; metrics are not collected\n"


# A dedicated registry keeps the app's metrics isolated and testable.
REGISTRY = CollectorRegistry()

# Default latency buckets (seconds) tuned for a low-latency pricing API.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUEST_COUNT = Counter(
    "pricing_http_requests_total",
    "Total HTTP requests processed.",
    labelnames=("method", "endpoint", "status"),
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "pricing_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "endpoint"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
REQUESTS_IN_PROGRESS = Gauge(
    "pricing_http_requests_in_progress",
    "Number of HTTP requests currently being served.",
    registry=REGISTRY,
)
PREDICTION_COUNT = Counter(
    "pricing_predictions_total",
    "Total price recommendations produced.",
    labelnames=("room_type", "model_version"),
    registry=REGISTRY,
)
MODEL_VERSION_INFO = Gauge(
    "pricing_model_loaded",
    "Set to 1 for the currently loaded model version label.",
    labelnames=("version",),
    registry=REGISTRY,
)
SCRAPE_COUNT = Counter(
    "pricing_scrape_runs_total",
    "Total competitor scrape attempts.",
    labelnames=("target", "status"),
    registry=REGISTRY,
)
ETL_RUNS = Counter(
    "pricing_etl_runs_total",
    "Total ETL pipeline runs.",
    labelnames=("status",),
    registry=REGISTRY,
)
TRAINING_RUNS = Counter(
    "pricing_training_runs_total",
    "Total model training runs.",
    labelnames=("status",),
    registry=REGISTRY,
)


def record_request(method: str, endpoint: str, status: int, duration_seconds: float) -> None:
    """Record a completed HTTP request's count and latency."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration_seconds)


def record_prediction(room_type: str, model_version: str) -> None:
    """Record a single produced price recommendation."""
    PREDICTION_COUNT.labels(room_type=room_type, model_version=model_version).inc()


def set_model_version(version: str) -> None:
    """Mark the currently loaded model version (sets the gauge to 1)."""
    MODEL_VERSION_INFO.labels(version=version).set(1.0)


def record_scrape(target: str, status: str) -> None:
    """Record a competitor scrape attempt outcome."""
    SCRAPE_COUNT.labels(target=target, status=status).inc()


def record_etl(status: str) -> None:
    """Record an ETL run outcome (``success`` or ``failure``)."""
    ETL_RUNS.labels(status=status).inc()


def record_training(status: str) -> None:
    """Record a training run outcome (``success`` or ``failure``)."""
    TRAINING_RUNS.labels(status=status).inc()


def render_latest() -> tuple[bytes, str]:
    """Return the metrics exposition payload and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
