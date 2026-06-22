"""Pricing service.

The application-facing orchestration for producing a price recommendation. It
ties together the demand model, the pricing engine, the business-rules layer,
and the explanation agent, and maps the internal :class:`PricingDecision` onto
the public :class:`PriceRecommendationResponse` schema.

Key guarantees:

* It builds features with :func:`default_feature_builder`, the *same* builder
  used during training, so serving-time features match training exactly.
* It is runnable with no trained model: :func:`load_demand_model` transparently
  falls back to the deterministic :class:`HeuristicDemandModel`, so the API
  produces sensible recommendations out of the box.
"""

from __future__ import annotations

from functools import lru_cache

from app.agents.explanation import ExplanationAgent
from app.agents.llm import get_llm_client
from app.config import Settings, get_settings
from app.feature_engineering.defaults import default_feature_builder
from app.inference import DemandModel, load_demand_model
from app.monitoring import record_prediction, set_model_version
from app.pricing import PricingEngine
from app.pricing.engine import PricingDecision
from app.schemas.pricing import (
    BusinessRules,
    PriceRecommendationRequest,
    PriceRecommendationResponse,
)
from app.utils.logging import get_logger

logger = get_logger("pricing")


class PricingService:
    """Stateful façade that produces price recommendations.

    The demand model, pricing engine, and explanation agent are constructed once
    per service instance. Use :func:`get_pricing_service` to obtain a cached,
    process-wide instance.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._feature_builder = default_feature_builder()
        self._model: DemandModel = load_demand_model(self._settings)
        self._engine = PricingEngine(self._model, self._feature_builder)
        self._explainer = ExplanationAgent(get_llm_client(self._settings))
        set_model_version(self._model.version)
        logger.info(
            "pricing service ready (model_version={})", self._model.version
        )

    @property
    def model_version(self) -> str:
        """Version string of the currently loaded demand model."""
        return self._model.version

    @property
    def model_available(self) -> bool:
        """Whether a trained model (as opposed to the heuristic) is loaded."""
        return not self._model.version.startswith("heuristic")

    def reload_model(self) -> str:
        """Reload the demand model from disk (e.g. after a training run)."""
        self._model = load_demand_model(self._settings)
        self._engine = PricingEngine(self._model, self._feature_builder)
        set_model_version(self._model.version)
        logger.info("pricing service reloaded model_version={}", self._model.version)
        return self._model.version

    def _to_response(
        self,
        request: PriceRecommendationRequest,
        decision: PricingDecision,
        rules: BusinessRules,
    ) -> PriceRecommendationResponse:
        """Map an internal decision onto the public response schema."""
        response = PriceRecommendationResponse(
            hotel_id=request.hotel_id,
            room_type=request.room_type,
            stay_date=request.stay_date,
            currency=rules.currency,
            recommended_price=decision.recommended_price,
            unconstrained_optimal_price=decision.unconstrained_optimal_price,
            expected_occupancy=decision.expected_occupancy,
            expected_revenue=decision.expected_revenue,
            effective_floor=decision.effective_floor,
            effective_ceiling=decision.effective_ceiling,
            price_change_pct=decision.price_change_pct,
            manual_override_applied=decision.manual_override_applied,
            applied_constraints=decision.applied_constraints,
            feature_drivers=decision.feature_drivers,
            model_version=decision.model_version,
        )
        if request.include_explanation:
            response.explanation = self._explainer.explain(
                request, decision, currency=rules.currency
            )
        return response

    def recommend(
        self, request: PriceRecommendationRequest
    ) -> PriceRecommendationResponse:
        """Produce a full price recommendation for a single request.

        Args:
            request: A self-contained recommendation request.

        Returns:
            The constraint-respecting :class:`PriceRecommendationResponse`,
            including an explanation when requested.
        """
        rules = request.business_rules or BusinessRules()
        decision = self._engine.recommend(request, rules)
        record_prediction(request.room_type.value, decision.model_version)
        logger.info(
            "recommended price hotel={} room={} stay={} price={:.2f} "
            "(optimal={:.2f}, model={})",
            request.hotel_id,
            request.room_type.value,
            request.stay_date.isoformat(),
            decision.recommended_price,
            decision.unconstrained_optimal_price,
            decision.model_version,
        )
        return self._to_response(request, decision, rules)


@lru_cache(maxsize=1)
def get_pricing_service() -> PricingService:
    """Return a process-wide cached :class:`PricingService` instance."""
    return PricingService(get_settings())
