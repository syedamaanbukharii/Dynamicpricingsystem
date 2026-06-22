"""Standalone explanation endpoint.

Returns the same recommendation as the pricing endpoint but is documented as the
explanation-focused entry point; the response always includes the natural-language
rationale regardless of the request's ``include_explanation`` flag.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_pricing_service_dependency, require_api_key
from app.schemas.pricing import (
    PriceExplanation,
    PriceRecommendationRequest,
)
from app.services.pricing_service import PricingService

router = APIRouter(tags=["pricing"], dependencies=[Depends(require_api_key)])


@router.post(
    "/explanations",
    response_model=PriceExplanation,
    summary="Explain a recommended price in business terms",
)
def explain_price(
    request: PriceRecommendationRequest,
    pricing: PricingService = Depends(get_pricing_service_dependency),
) -> PriceExplanation:
    """Return the business-friendly explanation for the recommended price."""
    request.include_explanation = True
    response = pricing.recommend(request)
    # ``recommend`` always populates an explanation when requested above.
    assert response.explanation is not None
    return response.explanation
