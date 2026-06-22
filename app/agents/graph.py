"""LangGraph orchestration of the competitor data pipeline.

Wires the cleaning agent (which internally performs room-name matching) and the
data-quality agent into a small, inspectable state graph:

    raw listings --> [clean] --> [quality assess] --> END

The graph produces structured :class:`CompetitorRate` rows plus a
:class:`QualityReport`. ``langgraph`` is imported here (a first-class project
dependency); the rest of :mod:`app.agents` does not import this module at
package-import time, so the individual agents remain usable even where the graph
runtime is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.competitor_cleaning import CompetitorCleaningAgent
from app.agents.data_quality import DataQualityAgent, QualityReport
from app.agents.llm import LLMClient
from app.config import Settings, get_settings
from app.schemas.competitor import CompetitorRate, RawCompetitorListing
from app.utils.logging import get_logger

logger = get_logger("agent")


class PipelineState(TypedDict, total=False):
    """Mutable state threaded through the competitor pipeline graph."""

    listings: list[RawCompetitorListing]
    default_stay_date: date | None
    rates: list[CompetitorRate]
    dropped: list[dict[str, str]]
    quality: QualityReport


@dataclass
class CompetitorPipelineResult:
    """Final result returned by :func:`run_competitor_pipeline`."""

    rates: list[CompetitorRate]
    dropped: list[dict[str, str]]
    quality: QualityReport


def build_competitor_pipeline(settings: Settings | None = None):
    """Compile and return the competitor cleaning + quality LangGraph."""
    settings = settings or get_settings()
    llm = LLMClient(settings)
    cleaner = CompetitorCleaningAgent(llm=llm)
    quality_agent = DataQualityAgent()

    def clean_node(state: PipelineState) -> PipelineState:
        result = cleaner.clean(
            state.get("listings", []),
            default_stay_date=state.get("default_stay_date"),
        )
        return {"rates": result.rates, "dropped": result.dropped}

    def quality_node(state: PipelineState) -> PipelineState:
        report = quality_agent.assess(state.get("rates", []))
        return {"quality": report}

    graph = StateGraph(PipelineState)
    graph.add_node("clean", clean_node)
    graph.add_node("quality", quality_node)
    graph.set_entry_point("clean")
    graph.add_edge("clean", "quality")
    graph.add_edge("quality", END)
    return graph.compile()


def run_competitor_pipeline(
    listings: list[RawCompetitorListing],
    *,
    settings: Settings | None = None,
    default_stay_date: date | None = None,
) -> CompetitorPipelineResult:
    """Run the full competitor pipeline and return structured results."""
    pipeline = build_competitor_pipeline(settings)
    final_state: PipelineState = pipeline.invoke(
        {"listings": listings, "default_stay_date": default_stay_date}
    )
    logger.info(
        "competitor pipeline complete: {} rates, {} dropped",
        len(final_state.get("rates", [])),
        len(final_state.get("dropped", [])),
    )
    return CompetitorPipelineResult(
        rates=final_state.get("rates", []),
        dropped=final_state.get("dropped", []),
        quality=final_state["quality"],
    )
