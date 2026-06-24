"""ETL pipeline endpoint (runs in the background)."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.api.deps import require_api_key, settings_dependency
from app.config import Settings
from app.schemas.common import JobAcceptedResponse
from app.services.etl_service import run_etl_pipeline
from app.utils.logging import get_logger

logger = get_logger("etl")

router = APIRouter(tags=["etl"], dependencies=[Depends(require_api_key)])


class ETLRequest(BaseModel):
    """Parameters controlling an ETL run."""

    default_stay_date: date | None = Field(
        default=None, description="Stay date for competitor listings missing one."
    )
    persist_to_db: bool = Field(
        default=True, description="Attempt to load results into PostgreSQL."
    )


def _run_etl_job(default_stay_date: date | None, persist_to_db: bool, settings: Settings) -> None:
    """Background worker that runs the ETL pipeline and logs the outcome."""
    try:
        summary = run_etl_pipeline(
            settings=settings,
            default_stay_date=default_stay_date,
            persist_to_db=persist_to_db,
        )
        logger.info("background ETL complete: {}", summary.get("load"))
    except Exception as exc:
        logger.exception("background ETL failed: {}", exc)


@router.post(
    "/etl/run",
    response_model=JobAcceptedResponse,
    status_code=202,
    summary="Trigger the ETL pipeline in the background",
)
def trigger_etl(
    request: ETLRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(settings_dependency),
) -> JobAcceptedResponse:
    """Accept an ETL request and run the pipeline in the background."""
    job_id = uuid.uuid4().hex
    background_tasks.add_task(
        _run_etl_job, request.default_stay_date, request.persist_to_db, settings
    )
    logger.info("accepted ETL job {}", job_id)
    return JobAcceptedResponse(job_id=job_id, job_type="etl", detail="ETL pipeline started.")
