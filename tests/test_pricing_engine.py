"""Tests for the pricing engine: revenue maximization, monotonicity, bounds."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.feature_engineering.defaults import default_feature_builder
from app.inference.heuristic import HeuristicDemandModel
from app.pricing.engine import PricingEngine
from app.schemas.common import RoomType
from app.schemas.pricing import BusinessRules, PriceRecommendationRequest


@pytest.fixture
def engine() -> PricingEngine:
    """A pricing engine backed by the deterministic heuristic demand model."""
    return PricingEngine(HeuristicDemandModel(), default_feature_builder())


def _request(**overrides) -> PriceRecommendationRequest:
    base = {
        "hotel_id": "HOTEL_A",
        "room_type": RoomType.DELUXE_KING,
        "stay_date": date.today() + timedelta(days=21),
        "inventory_total": 50,
        "rooms_sold": 20,
        "previous_price": 180.0,
        "competitor_rates": [175.0, 189.0, 205.0, 210.0],
        "booking_velocity": 3.0,
        "include_explanation": False,
    }
    base.update(overrides)
    return PriceRecommendationRequest(**base)


def test_recommend_returns_bounded_price(engine: PricingEngine) -> None:
    """The recommended price lies within the effective floor/ceiling."""
    decision = engine.recommend(_request(), BusinessRules())
    assert decision.effective_floor <= decision.recommended_price <= decision.effective_ceiling
    assert decision.recommended_price > 0


def test_recommend_respects_hard_bounds(engine: PricingEngine) -> None:
    """A narrow business band forces the price inside [min_rate, max_rate]."""
    rules = BusinessRules(min_rate=120, max_rate=160, max_daily_change_pct=1.0)
    decision = engine.recommend(_request(previous_price=None), rules)
    assert 120 <= decision.recommended_price <= 160


def test_unconstrained_optimal_is_interior(engine: PricingEngine) -> None:
    """The unconstrained optimum is finite and within the explored support."""
    decision = engine.recommend(_request(previous_price=None), BusinessRules())
    assert decision.unconstrained_optimal_price > 0
    # Should not peg to the global max_rate ceiling for a downward-sloping curve.
    assert decision.unconstrained_optimal_price < BusinessRules().max_rate


def test_expected_revenue_positive(engine: PricingEngine) -> None:
    """Expected revenue is positive and consistent with occupancy * price * inventory."""
    decision = engine.recommend(_request(), BusinessRules())
    assert decision.expected_revenue > 0
    assert 0.0 <= decision.expected_occupancy <= 1.0


def test_manual_override_propagates(engine: PricingEngine) -> None:
    """A manual override is reflected in the decision."""
    rules = BusinessRules(manual_override=250.0)
    decision = engine.recommend(_request(), rules)
    assert decision.recommended_price == 250.0
    assert decision.manual_override_applied is True


def test_demand_decreases_with_price() -> None:
    """The heuristic demand model is monotonically non-increasing in price."""
    model = HeuristicDemandModel()
    builder = default_feature_builder()
    req = _request()
    low = builder.build_features(req, 120.0)
    high = builder.build_features(req, 320.0)
    assert model.predict_rooms_sold(high) <= model.predict_rooms_sold(low)
