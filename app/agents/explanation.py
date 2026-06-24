"""Pricing Explanation Agent.

Produces a concise, business-friendly rationale for an already-computed pricing
decision. The LLM is used **only** to phrase an explanation of a decision the
XGBoost-driven engine already made -- never to make or alter the price. When the
LLM is disabled or unavailable, a deterministic rule-based summary is generated
from the decision's drivers and applied constraints, so an explanation is always
returned.
"""

from __future__ import annotations

from app.agents.llm import LLMClient
from app.pricing.engine import PricingDecision
from app.schemas.common import Currency
from app.schemas.pricing import PriceExplanation, PriceRecommendationRequest
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger

logger = get_logger("agent")


class ExplanationAgent:
    """Generates human-readable explanations for pricing decisions."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    @staticmethod
    def _driver_lines(request: PriceRecommendationRequest, decision: PricingDecision) -> list[str]:
        """Build the structured, factual driver bullet points."""
        drivers = decision.feature_drivers
        lines = [
            f"Occupancy is {drivers.get('occupancy', 0.0):.0%} with "
            f"{int(drivers.get('rooms_remaining', 0))} rooms remaining.",
            f"Booking pace (velocity) is {drivers.get('booking_velocity', 0.0):.2f} "
            "rooms/week-equivalent.",
            f"Lead time to arrival is {int(drivers.get('lead_time_days', 0))} days.",
        ]
        comp_median = drivers.get("competitor_median", 0.0)
        if comp_median > 0:
            lines.append(
                f"Competitor median is {comp_median:.2f} (spread "
                f"{drivers.get('competitor_spread', 0.0):.2f})."
            )
        lines.append(
            f"The demand model's revenue-optimal price was "
            f"{decision.unconstrained_optimal_price:.2f}."
        )
        lines.extend(
            f"Constraint applied: {constraint.description}"
            for constraint in decision.applied_constraints
        )
        return lines

    def _rule_based(
        self,
        request: PriceRecommendationRequest,
        decision: PricingDecision,
        currency: Currency,
    ) -> PriceExplanation:
        """Compose a deterministic explanation without the LLM."""
        change = (
            f" ({decision.price_change_pct:+.1%} vs the previous price)"
            if decision.price_change_pct is not None
            else ""
        )
        override = " A manual override was applied." if decision.manual_override_applied else ""
        summary = (
            f"Recommended {decision.recommended_price:.2f} {currency.value} for "
            f"{request.room_type.value} on {request.stay_date.isoformat()}{change}. "
            f"Expected occupancy at this price is {decision.expected_occupancy:.0%} "
            f"for projected revenue of {decision.expected_revenue:.2f} "
            f"{currency.value}.{override}"
        )
        return PriceExplanation(
            summary=summary,
            drivers=self._driver_lines(request, decision),
            generated_by="rule-based",
        )

    def explain(
        self,
        request: PriceRecommendationRequest,
        decision: PricingDecision,
        currency: Currency = Currency.USD,
    ) -> PriceExplanation:
        """Return a business-friendly explanation for a pricing decision."""
        fallback = self._rule_based(request, decision, currency)
        if self._llm is None or not self._llm.enabled:
            return fallback
        try:
            facts = "\n".join(f"- {line}" for line in fallback.drivers)
            summary = self._llm.complete(
                system=(
                    "You are a revenue-management analyst. Write a concise, "
                    "two-to-three sentence explanation of a hotel room price "
                    "recommendation for a non-technical manager. Do not invent "
                    "numbers; use only the facts provided. Do not suggest a "
                    "different price."
                ),
                user=(
                    f"Final recommended price: {decision.recommended_price:.2f} "
                    f"{currency.value} for {request.room_type.value} on "
                    f"{request.stay_date.isoformat()}.\n"
                    f"Expected occupancy: {decision.expected_occupancy:.0%}.\n"
                    f"Expected revenue: {decision.expected_revenue:.2f} "
                    f"{currency.value}.\nKey factors:\n{facts}"
                ),
            )
            return PriceExplanation(
                summary=summary,
                drivers=fallback.drivers,
                generated_by=f"claude:{self._llm.model}",
            )
        except LLMError as exc:
            logger.warning("explanation LLM call failed, using rule-based: {}", exc)
            return fallback
