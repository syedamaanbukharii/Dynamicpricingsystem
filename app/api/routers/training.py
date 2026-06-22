"""Model training endpoint (runs in the background)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.api.deps import (
    get_pricing_service_dependency,
    require_api_key,
    settings_dependency,
)
from app.config import Settings
from app.schemas.common import JobAcceptedResponse
from app.services.pricing_service import PricingService
from app.services.training_service import train_from_file
from app.utils.logging import get_logger

logger = get_logger("ml")

router = APIRouter(tags=["training"], dependencies=[Depends(require_api_key)])


class TrainingRequest(BaseModel):
    """Parameters controlling a training run."""

    data_path: str = Field(
        ..., description="Path to a CSV/Parquet observation file with a rooms_sold column."
    )
    val_fraction: float = Field(
        default=0.2, gt=0.0, lt=0.9, description="Chronological validation hold-out fraction."
    )
    reload_after: bool = Field(
        default=True, description="Reload the serving model once training completes."
    )


def _run_training_job(
    data_path: str,
    val_fraction: float,
    reload_after: bool,
    settings: Settings,
    pricing: PricingService,
) -> None:
    """Background worker that trains a model and optionally hot-reloads it."""
    try:
        result = train_from_file(
            data_path, settings=settings, val_fraction=val_fraction
        )
        logger.info(
            "background training complete version={} metrics={}",
            result.version,
            result.metrics,
        )
        if reload_after:
            pricing.reload_model()
    except Exception as exc:
        logger.exception("background training failed: {}", exc)


@router.post(
    "/training/run",
    response_model=JobAcceptedResponse,
    status_code=202,
    summary="Trigger model (re)training in the background",
)
def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(settings_dependency),
    pricing: PricingService = Depends(get_pricing_service_dependency),
) -> JobAcceptedResponse:
    """Accept a training request and run it in the background."""
    job_id = uuid.uuid4().hex
    background_tasks.add_task(
        _run_training_job,
        request.data_path,
        request.val_fraction,
        request.reload_after,
        settings,
        pricing,
    )
    logger.info("accepted training job {} for {}", job_id, request.data_path)
    return JobAcceptedResponse(
        job_id=job_id, job_type="training", detail="Model training started."
    )
