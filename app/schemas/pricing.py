"""Pricing domain schemas: rules, model feature vector, requests and responses."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import Currency, RoomType, Season, utcnow


class BusinessRules(BaseModel):
    """Configurable constraints applied on top of the model recommendation.

    Defaults are conservative and safe; production callers typically override
    them per hotel/room-type from the configuration service or database.
    """

    min_rate: float = Field(default=49.0, ge=0, description="Absolute price floor.")
    max_rate: float = Field(default=999.0, gt=0, description="Absolute price ceiling.")
    variable_cost_per_room: float = Field(
        default=25.0, ge=0, description="Marginal cost used for margin enforcement."
    )
    min_margin_pct: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Minimum gross margin over variable cost (0..1).",
    )
    max_daily_change_pct: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Maximum allowed move vs the previously published price.",
    )
    occupancy_floor: float = Field(
        default=0.40,
        ge=0,
        le=1,
        description="Below this occupancy, discount pressure is favored.",
    )
    occupancy_ceiling: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Above this occupancy, upward pressure is favored.",
    )
    manual_override: float | None = Field(
        default=None, ge=0, description="If set, forces the published price."
    )
    rounding_increment: float = Field(
        default=1.0, gt=0, description="Round final price to this increment."
    )
    currency: Currency = Currency.USD

    @model_validator(mode="after")
    def _check_bounds(self) -> BusinessRules:
        if self.min_rate > self.max_rate:
            raise ValueError("min_rate cannot exceed max_rate")
        if self.occupancy_floor > self.occupancy_ceiling:
            raise ValueError("occupancy_floor cannot exceed occupancy_ceiling")
        return self

    def margin_floor(self) -> float:
        """Lowest price that still satisfies the minimum-margin rule."""
        if self.min_margin_pct >= 1:
            return self.max_rate
        return self.variable_cost_per_room / (1.0 - self.min_margin_pct)


class PricingFeatures(BaseModel):
    """The engineered feature vector consumed by the demand model.

    ``price`` is intentionally part of the vector so the XGBoost model can learn
    price sensitivity; the pricing engine evaluates the model across a grid of
    candidate prices to find the revenue-maximizing point.
    """

    price: float = Field(ge=0)
    lead_time_days: int = Field(ge=0)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    season: Season
    is_weekend: bool
    is_holiday: bool
    days_to_holiday: int = Field(ge=0)
    is_event: bool
    event_score: float = Field(ge=0)
    current_occupancy: float = Field(ge=0, le=1)
    rooms_remaining: int = Field(ge=0)
    inventory_total: int = Field(ge=0)
    booking_velocity: float = Field(ge=0)
    competitor_median: float = Field(ge=0)
    competitor_min: float = Field(ge=0)
    competitor_max: float = Field(ge=0)
    competitor_spread: float = Field(ge=0)
    price_to_comp_median: float = Field(ge=0)
    recent_price_change_pct: float
    demand_score: float = Field(ge=0, le=1)
    cancellation_rate: float = Field(ge=0, le=1)

    #: Ordered list of model input columns; keep in sync with the trainer.
    FEATURE_ORDER: ClassVar[tuple[str, ...]] = (
        "price",
        "lead_time_days",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "days_to_holiday",
        "is_event",
        "event_score",
        "current_occupancy",
        "rooms_remaining",
        "inventory_total",
        "booking_velocity",
        "competitor_median",
        "competitor_min",
        "competitor_max",
        "competitor_spread",
        "price_to_comp_median",
        "recent_price_change_pct",
        "demand_score",
        "cancellation_rate",
    )

    def to_model_row(self) -> dict[str, float]:
        """Return the numeric model input row in canonical column order."""
        raw = self.model_dump()
        return {col: float(raw[col]) for col in self.FEATURE_ORDER}


class PriceRecommendationRequest(BaseModel):
    """Self-contained request for a single (room_type, stay_date) decision."""

    hotel_id: str
    room_type: RoomType
    stay_date: date
    as_of_date: date = Field(default_factory=lambda: utcnow().date())
    inventory_total: int = Field(ge=1)
    rooms_sold: int = Field(default=0, ge=0)
    previous_price: float | None = Field(default=None, ge=0)
    competitor_rates: list[float] | None = Field(
        default=None, description="Cleaned competitor nightly rates, if available."
    )
    booking_velocity: float | None = Field(default=None, ge=0)
    demand_score: float | None = Field(default=None, ge=0, le=1)
    cancellation_rate: float | None = Field(default=None, ge=0, le=1)
    is_holiday: bool | None = None
    is_event: bool | None = None
    business_rules: BusinessRules | None = None
    include_explanation: bool = True

    @model_validator(mode="after")
    def _check(self) -> PriceRecommendationRequest:
        if self.rooms_sold > self.inventory_total:
            raise ValueError("rooms_sold cannot exceed inventory_total")
        if self.stay_date < self.as_of_date:
            raise ValueError("stay_date cannot precede as_of_date")
        return self


class AppliedConstraint(BaseModel):
    """Records that a particular business rule bound the final price."""

    rule: str
    description: str


class PriceExplanation(BaseModel):
    """Business-friendly rationale generated by the Explanation Agent (LLM)."""

    summary: str
    drivers: list[str] = Field(default_factory=list)
    generated_by: str = "rule-based"


class PriceRecommendationResponse(BaseModel):
    """Final, constraint-respecting recommendation returned to the caller."""

    hotel_id: str
    room_type: RoomType
    stay_date: date
    currency: Currency
    recommended_price: float
    unconstrained_optimal_price: float
    expected_occupancy: float
    expected_revenue: float
    effective_floor: float
    effective_ceiling: float
    price_change_pct: float | None
    manual_override_applied: bool
    applied_constraints: list[AppliedConstraint] = Field(default_factory=list)
    feature_drivers: dict[str, float] = Field(default_factory=dict)
    model_version: str
    explanation: PriceExplanation | None = None
    generated_at: str = Field(default_factory=lambda: utcnow().isoformat())
