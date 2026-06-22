"""Demand-model interface shared by the XGBoost model and the heuristic fallback."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.pricing import PricingFeatures


@runtime_checkable
class DemandModel(Protocol):
    """Predicts expected rooms sold for a (price, context) feature row.

    The pricing engine evaluates this across a price grid to maximize expected
    revenue. Implementations must be deterministic for a given input so that
    recommendations are reproducible and auditable.
    """

    @property
    def version(self) -> str:
        """Human-readable model version/identifier."""
        ...

    def predict_rooms_sold(self, features: PricingFeatures) -> float:
        """Expected number of rooms sold for the stay date at this price."""
        ...

    def predict_many(self, rows: list[PricingFeatures]) -> list[float]:
        """Vectorized prediction for several candidate feature rows."""
        ...
