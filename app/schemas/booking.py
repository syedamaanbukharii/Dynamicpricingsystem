"""Schemas describing historical demand-side records."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import BookingChannel, Currency, RoomType


class BookingRecord(BaseModel):
    """A single historical reservation (post-cleaning, canonical form)."""

    hotel_id: str
    booking_id: str
    room_type: RoomType
    booking_date: date
    stay_date: date
    nights: int = Field(default=1, ge=1)
    rooms: int = Field(default=1, ge=1)
    price: float = Field(ge=0)
    currency: Currency = Currency.USD
    channel: BookingChannel = BookingChannel.DIRECT
    is_cancelled: bool = False

    @property
    def lead_time_days(self) -> int:
        """Days between booking and arrival (floored at zero)."""
        return max((self.stay_date - self.booking_date).days, 0)

    @model_validator(mode="after")
    def _check_dates(self) -> BookingRecord:
        if self.stay_date < self.booking_date:
            raise ValueError("stay_date cannot precede booking_date")
        return self


class OccupancySnapshot(BaseModel):
    """Occupancy and pace for a (hotel, room_type, stay_date) as of a moment."""

    hotel_id: str
    room_type: RoomType
    stay_date: date
    as_of: datetime
    inventory_total: int = Field(ge=0)
    rooms_sold: int = Field(ge=0)
    bookings_last_7d: int = Field(default=0, ge=0)

    @property
    def occupancy(self) -> float:
        """Fraction of inventory sold (0..1)."""
        if self.inventory_total == 0:
            return 0.0
        return min(self.rooms_sold / self.inventory_total, 1.0)

    @property
    def rooms_remaining(self) -> int:
        """Unsold inventory."""
        return max(self.inventory_total - self.rooms_sold, 0)

    @model_validator(mode="after")
    def _check_inventory(self) -> OccupancySnapshot:
        if self.rooms_sold > self.inventory_total:
            raise ValueError("rooms_sold cannot exceed inventory_total")
        return self


class PriceObservation(BaseModel):
    """A historical own-rate observation used to detect day-over-day swings."""

    hotel_id: str
    room_type: RoomType
    stay_date: date
    observed_on: date
    price: float = Field(ge=0)
    currency: Currency = Currency.USD
