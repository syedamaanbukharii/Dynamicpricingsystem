"""Schemas for competitor pricing -- raw (scraped) and cleaned forms."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.schemas.common import Currency, RoomType, utcnow


class RawCompetitorListing(BaseModel):
    """Unstructured listing as emitted by a scraper, before agent cleaning."""

    source: str
    competitor: str
    raw_room_name: str
    raw_price: str
    stay_date: date | None = None
    scraped_at: datetime = Field(default_factory=utcnow)
    url: str | None = None
    is_advertisement: bool = False


class CompetitorRate(BaseModel):
    """A cleaned, structured competitor rate (post agent normalization)."""

    source: str
    competitor: str
    room_type: RoomType
    raw_room_name: str
    price: float = Field(ge=0)
    currency: Currency = Currency.USD
    stay_date: date
    scraped_at: datetime = Field(default_factory=utcnow)


class CompetitorPriceStats(BaseModel):
    """Aggregated competitor statistics for a (room_type, stay_date)."""

    room_type: RoomType
    stay_date: date
    sample_size: int = Field(ge=0)
    median: float = Field(ge=0)
    minimum: float = Field(ge=0)
    maximum: float = Field(ge=0)
    mean: float = Field(ge=0)

    @property
    def spread(self) -> float:
        """Absolute spread between the cheapest and dearest competitor."""
        return max(self.maximum - self.minimum, 0.0)
