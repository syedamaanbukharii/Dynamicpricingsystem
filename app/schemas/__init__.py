"""Pydantic schema package (the public data contracts)."""

from app.schemas.booking import BookingRecord, OccupancySnapshot, PriceObservation
from app.schemas.common import (
    BookingChannel,
    Currency,
    ErrorResponse,
    HealthResponse,
    JobAcceptedResponse,
    RoomType,
    Season,
)
from app.schemas.competitor import (
    CompetitorPriceStats,
    CompetitorRate,
    RawCompetitorListing,
)
from app.schemas.events import Holiday, LocalEvent
from app.schemas.pricing import (
    AppliedConstraint,
    BusinessRules,
    PriceExplanation,
    PriceRecommendationRequest,
    PriceRecommendationResponse,
    PricingFeatures,
)

__all__ = [
    "AppliedConstraint",
    "BookingChannel",
    "BookingRecord",
    "BusinessRules",
    "CompetitorPriceStats",
    "CompetitorRate",
    "Currency",
    "ErrorResponse",
    "HealthResponse",
    "Holiday",
    "JobAcceptedResponse",
    "LocalEvent",
    "OccupancySnapshot",
    "PriceExplanation",
    "PriceObservation",
    "PriceRecommendationRequest",
    "PriceRecommendationResponse",
    "PricingFeatures",
    "RawCompetitorListing",
    "RoomType",
    "Season",
]
