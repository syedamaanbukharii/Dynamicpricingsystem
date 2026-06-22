"""API routers."""

from app.api.routers import (
    etl,
    explanation,
    health,
    metrics,
    prediction,
    scraping,
    training,
)

__all__ = [
    "etl",
    "explanation",
    "health",
    "metrics",
    "prediction",
    "scraping",
    "training",
]
