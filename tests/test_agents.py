"""Tests for the agents: room matching, cleaning, quality, explanation (offline)."""

from __future__ import annotations

from datetime import date, timedelta

from app.agents.competitor_cleaning import CompetitorCleaningAgent
from app.agents.data_quality import DataQualityAgent
from app.agents.explanation import ExplanationAgent
from app.agents.room_matching import RoomMatcher
from app.feature_engineering.defaults import default_feature_builder
from app.inference.heuristic import HeuristicDemandModel
from app.pricing.engine import PricingEngine
from app.schemas.common import RoomType
from app.schemas.pricing import BusinessRules, PriceRecommendationRequest


def test_room_matcher_canonicalizes() -> None:
    """Common variants and typos map to the right canonical room types."""
    matcher = RoomMatcher()
    assert matcher.match("Deluxe King Room") == RoomType.DELUXE_KING
    assert matcher.match("King Deluxe") == RoomType.DELUXE_KING
    assert matcher.match("Premium King Roo") == RoomType.DELUXE_KING
    assert matcher.match("Std. Queen") == RoomType.STANDARD_QUEEN
    assert matcher.match("Junior Suite") == RoomType.JUNIOR_SUITE
    assert matcher.match("Mystery Pod 9000") == RoomType.OTHER


def test_cleaning_drops_ads_and_unparseable(raw_listings) -> None:
    """Advertisements and unparseable prices are dropped; valid rates kept."""
    agent = CompetitorCleaningAgent()
    result = agent.clean(raw_listings, default_stay_date=date.today() + timedelta(days=5))
    assert result.kept >= 1
    assert result.removed >= 1  # at least the "Sponsored" ad and the n/a price
    for rate in result.rates:
        assert rate.price > 0
        assert isinstance(rate.room_type, RoomType)


def test_quality_report_ok_for_clean_rates(raw_listings) -> None:
    """A cleaned batch yields a quality report without errors."""
    cleaned = CompetitorCleaningAgent().clean(
        raw_listings, default_stay_date=date.today() + timedelta(days=5)
    )
    report = DataQualityAgent().assess(cleaned.rates)
    assert report.is_ok
    assert not report.has_errors


def test_explanation_is_rule_based_offline() -> None:
    """With the LLM disabled, the explanation is deterministic and rule-based."""
    engine = PricingEngine(HeuristicDemandModel(), default_feature_builder())
    request = PriceRecommendationRequest(
        hotel_id="HOTEL_A",
        room_type=RoomType.DELUXE_KING,
        stay_date=date.today() + timedelta(days=14),
        inventory_total=50,
        rooms_sold=30,
        previous_price=200.0,
        competitor_rates=[190.0, 205.0, 220.0],
    )
    decision = engine.recommend(request, BusinessRules())
    explanation = ExplanationAgent(llm=None).explain(request, decision)
    assert explanation.generated_by == "rule-based"
    assert explanation.summary
    assert len(explanation.drivers) >= 1
