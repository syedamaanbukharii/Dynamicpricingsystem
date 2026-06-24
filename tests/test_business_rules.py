"""Tests for the business-rules layer: floors, ceilings, swing, override, occupancy."""

from __future__ import annotations

from app.pricing.rules import apply_business_rules, round_to_increment
from app.schemas.pricing import BusinessRules


def test_hard_floor_enforced() -> None:
    """A price below the min rate is raised to the floor."""
    rules = BusinessRules(min_rate=80, max_rate=400)
    result = apply_business_rules(50.0, rules, previous_price=None, occupancy=0.5)
    assert result.price >= 80
    assert result.effective_floor >= 80


def test_hard_ceiling_enforced() -> None:
    """A price above the max rate is capped at the ceiling."""
    rules = BusinessRules(min_rate=80, max_rate=300)
    result = apply_business_rules(999.0, rules, previous_price=None, occupancy=0.5)
    assert result.price <= 300


def test_margin_floor_enforced() -> None:
    """The floor never drops below the variable-cost margin floor."""
    rules = BusinessRules(min_rate=10, max_rate=400, variable_cost_per_room=100, min_margin_pct=0.2)
    result = apply_business_rules(20.0, rules, previous_price=None, occupancy=0.5)
    # margin floor = 100 * 1.2 = 120
    assert result.price >= 120


def test_max_daily_increase_limited() -> None:
    """Up-moves are limited to the configured daily-change percentage."""
    rules = BusinessRules(min_rate=50, max_rate=999, max_daily_change_pct=0.2)
    result = apply_business_rules(500.0, rules, previous_price=100.0, occupancy=0.6)
    assert result.price <= 100 * 1.2 + 1e-6


def test_max_daily_decrease_limited() -> None:
    """Down-moves are limited to the configured daily-change percentage."""
    rules = BusinessRules(min_rate=10, max_rate=999, max_daily_change_pct=0.2)
    result = apply_business_rules(10.0, rules, previous_price=100.0, occupancy=0.6)
    assert result.price >= 100 * 0.8 - 1e-6


def test_manual_override_bypasses_optimization() -> None:
    """A manual override is honored (clamped only by hard bounds)."""
    rules = BusinessRules(min_rate=50, max_rate=400, manual_override=250.0)
    result = apply_business_rules(123.0, rules, previous_price=100.0, occupancy=0.6)
    assert result.price == 250.0
    assert result.manual_override_applied is True


def test_manual_override_clamped_to_hard_bounds() -> None:
    """An out-of-range override is clamped to the hard ceiling."""
    rules = BusinessRules(min_rate=50, max_rate=300, manual_override=10_000.0)
    result = apply_business_rules(123.0, rules, previous_price=None, occupancy=0.6)
    assert result.price <= 300


def test_low_occupancy_caps_price() -> None:
    """Occupancy at/below the floor caps price at the previous level."""
    rules = BusinessRules(min_rate=50, max_rate=999, occupancy_floor=0.4)
    result = apply_business_rules(400.0, rules, previous_price=180.0, occupancy=0.3)
    assert result.price <= 180.0


def test_high_occupancy_holds_price_up() -> None:
    """Occupancy at/above the ceiling holds price at or above the previous level."""
    rules = BusinessRules(min_rate=50, max_rate=999, occupancy_ceiling=0.85)
    result = apply_business_rules(60.0, rules, previous_price=180.0, occupancy=0.95)
    assert result.price >= 180.0 * (1 - rules.max_daily_change_pct) - 1e-6


def test_rounding_increment() -> None:
    """Prices are rounded to the configured increment."""
    assert round_to_increment(123.4, 1.0) == 123.0
    assert round_to_increment(123.6, 1.0) == 124.0
    assert round_to_increment(123.0, 5.0) == 125.0
    assert round_to_increment(121.0, 5.0) == 120.0
