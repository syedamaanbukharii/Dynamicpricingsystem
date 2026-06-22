"""A closed-form demand model used as a safe fallback.

This lets the platform produce sensible, monotonic recommendations before any
XGBoost model has been trained, and gives the pricing-engine unit tests a fast,
deterministic dependency. It encodes a downward-sloping demand curve anchored to
competitor pricing with bounded uplifts for events, weekends, and holidays.
"""

from __future__ import annotations

import math

from app.inference.base import DemandModel
from app.schemas.pricing import PricingFeatures


class HeuristicDemandModel(DemandModel):
    """Elasticity-based demand model (no training required)."""

    def __init__(self, elasticity: float = 1.6) -> None:
        self._elasticity = float(elasticity)

    @property
    def version(self) -> str:  # noqa: D102
        return "heuristic-1.0"

    def predict_rooms_sold(self, features: PricingFeatures) -> float:  # noqa: D102
        reference = (
            features.competitor_median
            if features.competitor_median > 0
            else features.price
        )
        ratio = features.price / reference if reference > 0 else 1.0
        base_fraction = 0.35 + 0.5 * features.demand_score
        fraction = base_fraction * math.exp(-self._elasticity * (ratio - 1.0))
        fraction *= 1.0 + 0.5 * features.event_score
        if features.is_weekend:
            fraction *= 1.05
        if features.is_holiday:
            fraction *= 1.10
        fraction *= 1.0 - 0.5 * features.cancellation_rate
        fraction = max(0.0, min(1.0, fraction))
        return fraction * float(features.inventory_total)

    def predict_many(self, rows: list[PricingFeatures]) -> list[float]:  # noqa: D102
        return [self.predict_rooms_sold(row) for row in rows]
