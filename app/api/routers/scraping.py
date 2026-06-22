"""Competitor scraping endpoint (runs in the background)."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.api.deps import require_api_key, settings_dependency
from app.config import Settings
from app.schemas.common import JobAcceptedResponse
from app.services.scraping_service import run_scrape
from app.utils.logging import get_logger

logger = get_logger("scraper")

router = APIRouter(tags=["scraping"], dependencies=[Depends(require_api_key)])


class ScrapeRequest(BaseModel):
    """Parameters controlling a scrape run."""

    stay_dates: list[date] = Field(..., min_length=1, description="Stay dates to scrape.")
    target_names: list[str] | None = Field(
        default=None, description="Restrict to these target names (default: all enabled)."
    )
    incremental: bool = Field(
        default=True, description="Skip dates scraped within the refresh window."
    )


def _run_scrape_job(
    stay_dates: list[date],
    target_names: list[str] | None,
    incremental: bool,
    settings: Settings,
) -> None:
    """Background worker that performs the scrape and logs the outcome."""
    try:
        result = run_scrape(
            stay_dates,
            settings=settings,
            target_names=target_names,
            incremental=incremental,
        )
        logger.info("background scrape complete: {}", result.to_dict())
    except Exception as exc:
        logger.exception("background scrape failed: {}", exc)


@router.post(
    "/scrape",
    response_model=JobAcceptedResponse,
    status_code=202,
    summary="Trigger a competitor scrape in the background",
)
def trigger_scrape(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(settings_dependency),
) -> JobAcceptedResponse:
    """Accept a scrape request and run it in the background."""
    job_id = uuid.uuid4().hex
    background_tasks.add_task(
        _run_scrape_job,
        request.stay_dates,
        request.target_names,
        request.incremental,
        settings,
    )
    logger.info("accepted scrape job {} for {} dates", job_id, len(request.stay_dates))
    return JobAcceptedResponse(
        job_id=job_id,
        job_type="scrape",
        detail=f"Scraping {len(request.stay_dates)} stay date(s).",
    )
