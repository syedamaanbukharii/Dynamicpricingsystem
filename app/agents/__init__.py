"""Agentic AI package (LLM-assisted cleaning, matching, quality, explanation).

The graph runtime (:mod:`app.agents.graph`) depends on ``langgraph`` and is
imported lazily via :func:`run_competitor_pipeline` so the individual agents can
be used wherever the graph runtime is not required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.competitor_cleaning import CleaningResult, CompetitorCleaningAgent
from app.agents.data_quality import (
    DataQualityAgent,
    QualityIssue,
    QualityReport,
    Severity,
)
from app.agents.explanation import ExplanationAgent
from app.agents.feature_engineering_agent import (
    FeatureEngineeringAgent,
    FeatureEngineeringResult,
)
from app.agents.llm import LLMClient, get_llm_client
from app.agents.room_matching import RoomMatcher

if TYPE_CHECKING:  # pragma: no cover
    from app.agents.graph import CompetitorPipelineResult


def run_competitor_pipeline(*args, **kwargs) -> CompetitorPipelineResult:
    """Lazily import and run the LangGraph competitor pipeline."""
    from app.agents.graph import run_competitor_pipeline as _run

    return _run(*args, **kwargs)


def build_competitor_pipeline(*args, **kwargs):
    """Lazily import and build the LangGraph competitor pipeline."""
    from app.agents.graph import build_competitor_pipeline as _build

    return _build(*args, **kwargs)


__all__ = [
    "CleaningResult",
    "CompetitorCleaningAgent",
    "DataQualityAgent",
    "ExplanationAgent",
    "FeatureEngineeringAgent",
    "FeatureEngineeringResult",
    "LLMClient",
    "QualityIssue",
    "QualityReport",
    "RoomMatcher",
    "Severity",
    "build_competitor_pipeline",
    "get_llm_client",
    "run_competitor_pipeline",
]
