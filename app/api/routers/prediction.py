"""Price recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_pricing_service_dependency, require_api_key
from app.schemas.pricing import (
    PriceRecommendationRequest,
    PriceRecommendationResponse,
)
from app.services.pricing_service import PricingService

router = APIRouter(tags=["pricing"], dependencies=[Depends(require_api_key)])


@router.post(
    "/recommendations",
    response_model=PriceRecommendationResponse,
    summary="Recommend a revenue-optimal nightly price",
)
def recommend_price(
    request: PriceRecommendationRequest,
    pricing: PricingService = Depends(get_pricing_service_dependency),
) -> PriceRecommendationResponse:
    """Produce a constraint-respecting price recommendation.

    The endpoint is fully functional without a trained model: the service falls
    back to a deterministic heuristic demand curve, so a freshly cloned, untrained
    deployment still returns sensible, bounded recommendations.
    """
    return pricing.recommend(request)
