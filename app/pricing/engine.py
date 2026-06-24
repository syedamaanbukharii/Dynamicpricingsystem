"""Revenue-optimizing pricing engine.

Given a demand model and a request, the engine searches a price grid to find the
revenue-maximizing nightly rate (expected revenue = price x expected rooms sold,
capped by inventory), then hands that optimum to the business-rules layer to
produce a publishable, swing-limited price. A two-stage (coarse then local)
search keeps the optimum accurate without an expensive dense grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.feature_engineering.features import FeatureBuilder
from app.inference.base import DemandModel
from app.pricing.rules import apply_business_rules
from app.schemas.pricing import (
    AppliedConstraint,
    BusinessRules,
    PriceRecommendationRequest,
)
from app.utils.exceptions import PricingError
from app.utils.logging import get_logger

logger = get_logger("pricing")


@dataclass
class PricingDecision:
    """Complete, auditable result of a pricing computation."""

    recommended_price: float
    unconstrained_optimal_price: float
    expected_occupancy: float
    expected_revenue: float
    effective_floor: float
    effective_ceiling: float
    price_change_pct: float | None
    manual_override_applied: bool
    model_version: str
    applied_constraints: list[AppliedConstraint] = field(default_factory=list)
    feature_drivers: dict[str, float] = field(default_factory=dict)


class PricingEngine:
    """Combines a demand model with business rules to recommend a price."""

    def __init__(
        self,
        model: DemandModel,
        feature_builder: FeatureBuilder | None = None,
        *,
        coarse_points: int = 41,
        refine_points: int = 21,
        exploration_floor_factor: float = 0.5,
        exploration_ceiling_factor: float = 1.8,
    ) -> None:
        if coarse_points < 3 or refine_points < 3:
            raise ValueError("grid sizes must be >= 3")
        if not 0.0 < exploration_floor_factor < 1.0:
            raise ValueError("exploration_floor_factor must be in (0, 1)")
        if exploration_ceiling_factor <= 1.0:
            raise ValueError("exploration_ceiling_factor must be > 1")
        self.model = model
        self.features = feature_builder or FeatureBuilder()
        self.coarse_points = coarse_points
        self.refine_points = refine_points
        self.exploration_floor_factor = exploration_floor_factor
        self.exploration_ceiling_factor = exploration_ceiling_factor

    def _expected_revenue(
        self, request: PriceRecommendationRequest, price: float
    ) -> tuple[float, float]:
        """Return (expected_rooms_sold, expected_revenue) at ``price``."""
        feats = self.features.build_features(request, price)
        rooms = self.model.predict_rooms_sold(feats)
        rooms = float(max(0.0, min(rooms, request.inventory_total)))
        return rooms, price * rooms

    def _search(
        self, request: PriceRecommendationRequest, low: float, high: float, points: int
    ) -> tuple[float, float, float]:
        """Grid-search [low, high]; return (best_price, best_rooms, best_revenue)."""
        grid = np.linspace(low, high, points)
        best_price, best_rooms, best_rev = float(grid[0]), 0.0, -1.0
        for price in grid:
            rooms, revenue = self._expected_revenue(request, float(price))
            if revenue > best_rev:
                best_price, best_rooms, best_rev = float(price), rooms, revenue
        return best_price, best_rooms, best_rev

    def recommend(
        self, request: PriceRecommendationRequest, rules: BusinessRules | None = None
    ) -> PricingDecision:
        """Compute the revenue-optimal, rule-respecting price for the request."""
        rules = rules or BusinessRules()
        stats = self.features.competitor_stats(
            request.competitor_rates or [], request.room_type, request.stay_date
        )

        # Constrain the price search to the region where the demand model is
        # trustworthy. Tree models extrapolate flat beyond their training
        # support, so probing far above observed competitor/previous prices
        # would yield a spurious "higher price is always better" optimum. We
        # anchor exploration to competitor context (falling back to the previous
        # price, then the midpoint of the business band) and intersect it with
        # the business-allowed range. Hard floors/ceilings are still enforced by
        # the rules layer afterwards.
        anchor = (
            stats.median
            or stats.maximum
            or request.previous_price
            or (rules.min_rate + rules.max_rate) / 2.0
        )
        explore_low = max(rules.min_rate * 0.5, anchor * self.exploration_floor_factor)
        explore_high = anchor * self.exploration_ceiling_factor
        search_low = max(1.0, min(explore_low, rules.max_rate))
        search_high = min(rules.max_rate, max(explore_high, anchor * 1.05))
        search_high = max(search_high, search_low + 1.0)
        if search_high <= search_low:
            raise PricingError(
                "Invalid price search window.",
                details={"low": search_low, "high": search_high},
            )

        coarse_price, _, _ = self._search(request, search_low, search_high, self.coarse_points)
        step = (search_high - search_low) / (self.coarse_points - 1)
        fine_low = max(search_low, coarse_price - step)
        fine_high = min(search_high, coarse_price + step)
        optimal_price, _, _ = self._search(request, fine_low, fine_high, self.refine_points)

        occupancy = request.rooms_sold / request.inventory_total if request.inventory_total else 0.0
        rule_result = apply_business_rules(
            optimal_price,
            rules,
            previous_price=request.previous_price,
            occupancy=occupancy,
        )

        final_rooms, final_revenue = self._expected_revenue(request, rule_result.price)
        expected_occupancy = (
            final_rooms / request.inventory_total if request.inventory_total else 0.0
        )

        change_pct: float | None = None
        if request.previous_price and request.previous_price > 0:
            change_pct = round(
                (rule_result.price - request.previous_price) / request.previous_price, 4
            )

        drivers = self._drivers(request, stats, optimal_price)
        logger.info(
            "priced hotel={} room={} stay={} -> {:.2f} (optimal {:.2f}, occ {:.0%})",
            request.hotel_id,
            request.room_type.value,
            request.stay_date.isoformat(),
            rule_result.price,
            optimal_price,
            expected_occupancy,
        )

        return PricingDecision(
            recommended_price=rule_result.price,
            unconstrained_optimal_price=round(optimal_price, 2),
            expected_occupancy=round(expected_occupancy, 4),
            expected_revenue=round(final_revenue, 2),
            effective_floor=rule_result.effective_floor,
            effective_ceiling=rule_result.effective_ceiling,
            price_change_pct=change_pct,
            manual_override_applied=rule_result.manual_override_applied,
            model_version=self.model.version,
            applied_constraints=rule_result.applied_constraints,
            feature_drivers=drivers,
        )

    @staticmethod
    def _drivers(
        request: PriceRecommendationRequest, stats, optimal_price: float
    ) -> dict[str, float]:
        """Compact, human-meaningful subset of features for the explanation layer."""
        occupancy = request.rooms_sold / request.inventory_total if request.inventory_total else 0.0
        return {
            "occupancy": round(occupancy, 4),
            "rooms_remaining": float(max(request.inventory_total - request.rooms_sold, 0)),
            "competitor_median": round(stats.median, 2),
            "competitor_spread": round(stats.spread, 2),
            "booking_velocity": round(request.booking_velocity or 0.0, 3),
            "lead_time_days": float(max((request.stay_date - request.as_of_date).days, 0)),
            "model_optimal_price": round(optimal_price, 2),
        }
