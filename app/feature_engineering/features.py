"""Feature engineering shared by training and inference.

The :class:`FeatureBuilder` produces the exact feature set declared by
:data:`PricingFeatures.FEATURE_ORDER`, guaranteeing that the columns a model is
trained on match the columns presented at inference. It offers two entry points:

* :meth:`build_features` -- one :class:`PricingFeatures` row for a candidate
  price (used by the pricing engine's revenue search).
* :meth:`build_feature_matrix` -- a vectorized pandas DataFrame for training.
"""

from __future__ import annotations

import statistics
from datetime import date

import numpy as np
import pandas as pd

from app.feature_engineering.calendars import EventCalendar, HolidayCalendar, season_of
from app.schemas.common import RoomType
from app.schemas.competitor import CompetitorPriceStats
from app.schemas.pricing import PriceRecommendationRequest, PricingFeatures
from app.utils.logging import get_logger

logger = get_logger("features")

_EPS = 1e-9


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising on a zero denominator."""
    return float(numerator) / float(denominator) if abs(denominator) > _EPS else 0.0


class FeatureBuilder:
    """Builds model-ready features from domain inputs."""

    def __init__(
        self,
        holidays: HolidayCalendar | None = None,
        events: EventCalendar | None = None,
    ) -> None:
        self.holidays = holidays or HolidayCalendar()
        self.events = events or EventCalendar()

    # -- Competitor aggregation -----------------------------------------
    @staticmethod
    def competitor_stats(
        rates: list[float], room_type: RoomType, stay_date: date
    ) -> CompetitorPriceStats:
        """Aggregate a list of competitor rates into robust statistics."""
        clean = [float(r) for r in rates if r is not None and r > 0]
        if not clean:
            return CompetitorPriceStats(
                room_type=room_type,
                stay_date=stay_date,
                sample_size=0,
                median=0.0,
                minimum=0.0,
                maximum=0.0,
                mean=0.0,
            )
        return CompetitorPriceStats(
            room_type=room_type,
            stay_date=stay_date,
            sample_size=len(clean),
            median=float(statistics.median(clean)),
            minimum=float(min(clean)),
            maximum=float(max(clean)),
            mean=float(statistics.fmean(clean)),
        )

    @staticmethod
    def derive_demand_score(
        occupancy: float, booking_velocity: float, event_score: float, is_holiday: bool
    ) -> float:
        """Heuristic 0..1 demand score when no external signal is supplied.

        Combines occupancy, normalized booking pace, event intensity, and a
        holiday bump. Used only as a fallback feature value; the model learns the
        true relationship from history.
        """
        pace = min(booking_velocity / 5.0, 1.0)
        holiday_bump = 0.1 if is_holiday else 0.0
        score = 0.5 * occupancy + 0.3 * pace + 0.15 * event_score + holiday_bump
        return float(max(0.0, min(1.0, score)))

    # -- Single-row (inference) -----------------------------------------
    def build_features(self, request: PriceRecommendationRequest, price: float) -> PricingFeatures:
        """Build a :class:`PricingFeatures` row for one candidate ``price``."""
        stay = request.stay_date
        lead_time = max((stay - request.as_of_date).days, 0)
        occupancy = _safe_div(request.rooms_sold, request.inventory_total)
        rooms_remaining = max(request.inventory_total - request.rooms_sold, 0)

        is_holiday = (
            request.is_holiday if request.is_holiday is not None else self.holidays.is_holiday(stay)
        )
        is_event = request.is_event if request.is_event is not None else self.events.is_event(stay)
        event_score = self.events.event_score(stay)
        days_to_holiday = self.holidays.days_to_next_holiday(stay)

        stats = self.competitor_stats(request.competitor_rates or [], request.room_type, stay)
        booking_velocity = request.booking_velocity or 0.0
        demand_score = (
            request.demand_score
            if request.demand_score is not None
            else self.derive_demand_score(
                occupancy, booking_velocity, event_score, bool(is_holiday)
            )
        )
        recent_change = _safe_div(
            price - (request.previous_price or price),
            request.previous_price or price,
        )

        return PricingFeatures(
            price=float(price),
            lead_time_days=lead_time,
            day_of_week=stay.weekday(),
            month=stay.month,
            season=season_of(stay.month),
            is_weekend=stay.weekday() >= 5,
            is_holiday=bool(is_holiday),
            days_to_holiday=days_to_holiday,
            is_event=bool(is_event),
            event_score=event_score,
            current_occupancy=occupancy,
            rooms_remaining=rooms_remaining,
            inventory_total=request.inventory_total,
            booking_velocity=booking_velocity,
            competitor_median=stats.median,
            competitor_min=stats.minimum,
            competitor_max=stats.maximum,
            competitor_spread=stats.spread,
            price_to_comp_median=_safe_div(price, stats.median) if stats.median else 1.0,
            recent_price_change_pct=recent_change,
            demand_score=demand_score,
            cancellation_rate=request.cancellation_rate or 0.0,
        )

    # -- Vectorized (training) ------------------------------------------
    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized feature construction for a historical observation frame.

        Expected input columns: ``stay_date`` (datetime-like), ``booking_date``
        (optional), ``price``, ``inventory_total``, ``rooms_sold``,
        ``previous_price`` (optional), ``competitor_median``, ``competitor_min``,
        ``competitor_max`` (optional), ``booking_velocity`` (optional),
        ``cancellation_rate`` (optional). Holiday/event flags are derived from the
        builder's calendars unless already present.
        """
        out = df.copy()
        stay = pd.to_datetime(out["stay_date"])
        booking = pd.to_datetime(out["booking_date"]) if "booking_date" in out.columns else stay

        out["lead_time_days"] = (stay - booking).dt.days.clip(lower=0)
        out["day_of_week"] = stay.dt.dayofweek
        out["month"] = stay.dt.month
        out["is_weekend"] = (stay.dt.dayofweek >= 5).astype(int)

        stay_dates = stay.dt.date
        if "is_holiday" not in out.columns:
            out["is_holiday"] = stay_dates.map(self.holidays.is_holiday).astype(int)
        out["days_to_holiday"] = stay_dates.map(self.holidays.days_to_next_holiday)
        if "is_event" not in out.columns:
            out["is_event"] = stay_dates.map(self.events.is_event).astype(int)
        if "event_score" not in out.columns:
            out["event_score"] = stay_dates.map(self.events.event_score)

        inv = out["inventory_total"].replace(0, np.nan)
        out["current_occupancy"] = (out["rooms_sold"] / inv).fillna(0.0).clip(0, 1)
        out["rooms_remaining"] = (out["inventory_total"] - out["rooms_sold"]).clip(lower=0)

        for col, default in (
            ("competitor_median", 0.0),
            ("competitor_min", 0.0),
            ("competitor_max", 0.0),
            ("booking_velocity", 0.0),
            ("cancellation_rate", 0.0),
        ):
            if col not in out.columns:
                out[col] = default
            out[col] = out[col].fillna(default)

        out["competitor_spread"] = (out["competitor_max"] - out["competitor_min"]).clip(lower=0)
        comp_median = out["competitor_median"].replace(0, np.nan)
        out["price_to_comp_median"] = (out["price"] / comp_median).fillna(1.0)

        if "previous_price" in out.columns:
            prev = out["previous_price"].replace(0, np.nan)
            out["recent_price_change_pct"] = ((out["price"] - prev) / prev).fillna(0.0)
        else:
            out["recent_price_change_pct"] = 0.0

        if "demand_score" not in out.columns:
            pace = (out["booking_velocity"] / 5.0).clip(upper=1.0)
            out["demand_score"] = (
                0.5 * out["current_occupancy"]
                + 0.3 * pace
                + 0.15 * out["event_score"]
                + 0.1 * out["is_holiday"]
            ).clip(0, 1)

        feature_cols = list(PricingFeatures.FEATURE_ORDER)
        missing = [c for c in feature_cols if c not in out.columns]
        if missing:
            raise ValueError(f"feature matrix missing columns: {missing}")
        logger.debug("built feature matrix: {} rows x {} cols", len(out), len(feature_cols))
        return out[feature_cols].astype(float)


def feature_columns() -> list[str]:
    """Return the canonical ordered model feature columns."""
    return list(PricingFeatures.FEATURE_ORDER)
