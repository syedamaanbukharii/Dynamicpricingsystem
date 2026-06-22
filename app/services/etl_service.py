"""ETL service.

A thin wrapper over the ETL pipeline (:func:`app.etl.run_etl`) that records a
Prometheus metric for the run outcome and returns the pipeline summary. Used by
the API's ETL router and the ETL CLI.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from app.config import Settings, get_settings
from app.etl import run_etl
from app.etl.extract import DataSource
from app.monitoring import record_etl
from app.utils.exceptions import ETLError
from app.utils.logging import get_logger

logger = get_logger("etl")


def run_etl_pipeline(
    *,
    settings: Settings | None = None,
    source: DataSource | None = None,
    default_stay_date: date | None = None,
    persist_to_db: bool = True,
) -> dict[str, Any]:
    """Run the ETL pipeline and record the outcome metric.

    Args:
        settings: Application settings (defaults to the cached singleton).
        source: Optional data source override (defaults to the file source).
        default_stay_date: Stay date for competitor listings missing one.
        persist_to_db: Whether to attempt the PostgreSQL load.

    Returns:
        The pipeline summary dictionary.
    """
    settings = settings or get_settings()
    try:
        summary = run_etl(
            source=source,
            settings=settings,
            default_stay_date=default_stay_date,
            persist_to_db=persist_to_db,
        )
    except Exception as exc:
        record_etl("failure")
        if isinstance(exc, ETLError):
            raise
        raise ETLError("ETL pipeline failed.", details={"error": str(exc)}) from exc

    record_etl("success")
    logger.info("ETL pipeline finished: {}", summary.get("load"))
    return summary
