"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_pricing_service_dependency, settings_dependency
from app.config import Settings
from app.schemas.common import HealthResponse
from app.services.pricing_service import PricingService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness/readiness probe")
def health(
    settings: Settings = Depends(settings_dependency),
    pricing: PricingService = Depends(get_pricing_service_dependency),
) -> HealthResponse:
    """Return service status, version, environment, and model availability."""
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment.value,
        model_available=pricing.model_available,
    )
