"""Shared enumerations and generic response envelopes."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class RoomType(str, Enum):
    """Canonical room-type vocabulary produced by the Room Matching Agent."""

    STANDARD_QUEEN = "STANDARD_QUEEN"
    STANDARD_KING = "STANDARD_KING"
    STANDARD_TWIN = "STANDARD_TWIN"
    DELUXE_QUEEN = "DELUXE_QUEEN"
    DELUXE_KING = "DELUXE_KING"
    JUNIOR_SUITE = "JUNIOR_SUITE"
    EXECUTIVE_SUITE = "EXECUTIVE_SUITE"
    FAMILY_ROOM = "FAMILY_ROOM"
    ACCESSIBLE = "ACCESSIBLE"
    OTHER = "OTHER"


class Currency(str, Enum):
    """Subset of supported ISO-4217 currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    INR = "INR"


class Season(str, Enum):
    """Meteorological season buckets (Northern-hemisphere convention)."""

    WINTER = "WINTER"
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    AUTUMN = "AUTUMN"


class BookingChannel(str, Enum):
    """Channel through which a booking was made."""

    DIRECT = "DIRECT"
    OTA = "OTA"
    GDS = "GDS"
    PHONE = "PHONE"
    WALK_IN = "WALK_IN"


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp."""
    return datetime.now(UTC)


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = "ok"
    version: str
    environment: str
    model_available: bool
    timestamp: datetime = Field(default_factory=utcnow)


class ErrorResponse(BaseModel):
    """Uniform error body returned by the exception middleware."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
    request_id: str | None = None


class JobAcceptedResponse(BaseModel):
    """Returned by endpoints that kick off background work."""

    job_id: str
    job_type: str
    status: str = "accepted"
    detail: str | None = None
    submitted_at: datetime = Field(default_factory=utcnow)
