"""Tests for Pydantic schema validators and derived properties."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from app.schemas.booking import BookingRecord, OccupancySnapshot
from app.schemas.common import RoomType
from app.schemas.pricing import BusinessRules, PriceRecommendationRequest
from pydantic import ValidationError as PydanticValidationError


def test_booking_rejects_stay_before_booking() -> None:
    """A stay date earlier than the booking date is rejected."""
    with pytest.raises(PydanticValidationError):
        BookingRecord(
            hotel_id="H",
            booking_id="B1",
            room_type=RoomType.STANDARD_QUEEN,
            booking_date=date(2026, 5, 10),
            stay_date=date(2026, 5, 1),
            price=100.0,
        )


def test_booking_lead_time_days() -> None:
    """Lead time is the non-negative day delta between booking and stay."""
    record = BookingRecord(
        hotel_id="H",
        booking_id="B1",
        room_type=RoomType.STANDARD_QUEEN,
        booking_date=date(2026, 5, 1),
        stay_date=date(2026, 5, 15),
        price=100.0,
    )
    assert record.lead_time_days == 14


def test_occupancy_rejects_oversell() -> None:
    """Rooms sold cannot exceed inventory."""
    with pytest.raises(PydanticValidationError):
        OccupancySnapshot(
            hotel_id="H",
            room_type=RoomType.DELUXE_KING,
            stay_date=date(2026, 5, 15),
            as_of=datetime.now(UTC),
            inventory_total=10,
            rooms_sold=20,
        )


def test_occupancy_properties() -> None:
    """Occupancy fraction and rooms-remaining are computed correctly."""
    snapshot = OccupancySnapshot(
        hotel_id="H",
        room_type=RoomType.DELUXE_KING,
        stay_date=date(2026, 5, 15),
        as_of=datetime.now(UTC),
        inventory_total=40,
        rooms_sold=10,
    )
    assert snapshot.occupancy == 0.25
    assert snapshot.rooms_remaining == 30


def test_recommendation_request_rejects_past_stay() -> None:
    """A stay date before the as-of date is rejected."""
    with pytest.raises(PydanticValidationError):
        PriceRecommendationRequest(
            hotel_id="H",
            room_type=RoomType.DELUXE_KING,
            stay_date=date.today() - timedelta(days=1),
            as_of_date=date.today(),
            inventory_total=10,
        )


def test_recommendation_request_defaults() -> None:
    """Sensible defaults are applied to an otherwise minimal request."""
    request = PriceRecommendationRequest(
        hotel_id="H",
        room_type=RoomType.DELUXE_KING,
        stay_date=date.today() + timedelta(days=10),
        inventory_total=20,
    )
    assert request.rooms_sold == 0
    assert request.include_explanation is True
    assert request.competitor_rates is None


def test_business_rules_margin_floor() -> None:
    """The margin floor is the price at which profit equals the margin % of price."""
    rules = BusinessRules(variable_cost_per_room=100.0, min_margin_pct=0.25)
    # margin as a fraction of selling price: price = cost / (1 - margin_pct)
    assert rules.margin_floor() == pytest.approx(100.0 / 0.75)


def test_business_rules_defaults_are_feasible() -> None:
    """Default rules form a feasible (floor <= ceiling) band."""
    rules = BusinessRules()
    assert rules.min_rate < rules.max_rate
    assert 0.0 < rules.max_daily_change_pct < 1.0
