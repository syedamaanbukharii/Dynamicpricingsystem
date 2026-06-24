"""SQLAlchemy 2.0 ORM models for persisted entities.

Using the typed declarative ORM means every query is parameterized, eliminating
SQL-injection risk by construction. Tables mirror the domain schemas: bookings,
price observations, occupancy, inventory, competitor rates, events, holidays,
emitted recommendations, and model-training runs.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class TimestampMixin:
    """Adds a server-side ``created_at`` column."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Booking(Base, TimestampMixin):
    """A historical reservation."""

    __tablename__ = "bookings"
    __table_args__ = (
        UniqueConstraint("hotel_id", "booking_id", name="uq_booking_identity"),
        Index("ix_bookings_lookup", "hotel_id", "room_type", "stay_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    booking_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    booking_date: Mapped[date] = mapped_column(Date, nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    nights: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    rooms: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    channel: Mapped[str] = mapped_column(String(16), default="DIRECT", nullable=False)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class PriceObservation(Base, TimestampMixin):
    """A historical own-rate observation."""

    __tablename__ = "price_observations"
    __table_args__ = (Index("ix_prices_lookup", "hotel_id", "room_type", "stay_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)


class OccupancySnapshot(Base, TimestampMixin):
    """Occupancy and booking pace as of a moment in time."""

    __tablename__ = "occupancy_snapshots"
    __table_args__ = (Index("ix_occ_lookup", "hotel_id", "room_type", "stay_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    inventory_total: Mapped[int] = mapped_column(Integer, nullable=False)
    rooms_sold: Mapped[int] = mapped_column(Integer, nullable=False)
    bookings_last_7d: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Inventory(Base, TimestampMixin):
    """Sellable inventory for a (hotel, room_type, stay_date)."""

    __tablename__ = "inventory"
    __table_args__ = (
        UniqueConstraint("hotel_id", "room_type", "stay_date", name="uq_inventory_slot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    inventory_total: Mapped[int] = mapped_column(Integer, nullable=False)


class CompetitorRate(Base, TimestampMixin):
    """A cleaned competitor rate."""

    __tablename__ = "competitor_rates"
    __table_args__ = (Index("ix_comp_lookup", "room_type", "stay_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    competitor: Mapped[str] = mapped_column(String(128), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_room_name: Mapped[str] = mapped_column(String(256), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Event(Base, TimestampMixin):
    """A demand-affecting local event."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    venue: Mapped[str | None] = mapped_column(String(256), nullable=True)
    expected_attendance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    demand_multiplier: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)


class Holiday(Base, TimestampMixin):
    """A public or observed holiday."""

    __tablename__ = "holidays"
    __table_args__ = (UniqueConstraint("holiday_date", "country", name="uq_holiday_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    holiday_date: Mapped[date] = mapped_column(Date, nullable=False)
    country: Mapped[str] = mapped_column(String(8), default="US", nullable=False)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Recommendation(Base, TimestampMixin):
    """A pricing recommendation emitted by the engine, for audit/history."""

    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_reco_lookup", "hotel_id", "room_type", "stay_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    room_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    recommended_price: Mapped[float] = mapped_column(Float, nullable=False)
    unconstrained_optimal_price: Mapped[float] = mapped_column(Float, nullable=False)
    expected_occupancy: Mapped[float] = mapped_column(Float, nullable=False)
    expected_revenue: Mapped[float] = mapped_column(Float, nullable=False)
    effective_floor: Mapped[float] = mapped_column(Float, nullable=False)
    effective_ceiling: Mapped[float] = mapped_column(Float, nullable=False)
    price_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    manual_override_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    explanation: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelRun(Base, TimestampMixin):
    """A record of a model-training run and its evaluation metrics."""

    __tablename__ = "model_runs"
    __table_args__ = (UniqueConstraint("version", name="uq_model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False)
    n_val: Mapped[int] = mapped_column(Integer, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    feature_importance: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    model_path: Mapped[str] = mapped_column(String(512), nullable=False)
