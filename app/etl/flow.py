"""ETL orchestration.

Wires extract -> transform -> load into a single pipeline. Prefect is used for
orchestration (retries, observability, scheduling) when installed, but the
import is guarded: if Prefect is unavailable, lightweight no-op ``task``/``flow``
decorators are substituted so the very same functions run as plain Python. This
keeps the pipeline fully runnable in minimal environments while remaining a
first-class Prefect flow in production.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, TypeVar

from app.config import Settings, get_settings
from app.etl.extract import DataSource, ExtractedData, extract
from app.etl.load import LoadResult, load
from app.etl.transform import TransformedData, transform
from app.utils.logging import get_logger

logger = get_logger("etl")

F = TypeVar("F", bound=Callable[..., Any])

try:  # pragma: no cover - exercised only when Prefect is installed
    from prefect import flow as _prefect_flow
    from prefect import task as _prefect_task

    PREFECT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path in minimal envs
    PREFECT_AVAILABLE = False

    def _identity_decorator(*dargs: Any, **dkwargs: Any) -> Any:
        """Return a decorator that leaves the wrapped callable unchanged."""

        def wrap(func: F) -> F:
            return func

        # Support both ``@task`` and ``@task(...)`` usage.
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return wrap

    _prefect_task = _identity_decorator  # type: ignore[assignment]
    _prefect_flow = _identity_decorator  # type: ignore[assignment]


@_prefect_task(name="etl-extract")
def extract_task(
    source: DataSource | None, settings: Settings
) -> ExtractedData:
    """Prefect task wrapping the extraction stage."""
    return extract(source, settings=settings)


@_prefect_task(name="etl-transform")
def transform_task(
    extracted: ExtractedData,
    settings: Settings,
    default_stay_date: date | None,
) -> TransformedData:
    """Prefect task wrapping the transformation stage."""
    return transform(
        extracted, settings=settings, default_stay_date=default_stay_date
    )


@_prefect_task(name="etl-load")
def load_task(
    transformed: TransformedData, settings: Settings, persist_to_db: bool
) -> LoadResult:
    """Prefect task wrapping the load stage."""
    return load(transformed, settings=settings, persist_to_db=persist_to_db)


@_prefect_flow(name="dynamic-pricing-etl")
def etl_flow(
    *,
    source: DataSource | None = None,
    settings: Settings | None = None,
    default_stay_date: date | None = None,
    persist_to_db: bool = True,
) -> dict[str, Any]:
    """Run the full ETL pipeline and return a summary.

    Args:
        source: Optional data source (defaults to a file source over
            ``settings.raw_data_dir``).
        settings: Application settings (defaults to the cached singleton).
        default_stay_date: Stay date for competitor listings missing one.
        persist_to_db: Whether to attempt the PostgreSQL load.

    Returns:
        A JSON-serializable summary dict of extracted counts, transform
        metadata, and load results.
    """
    settings = settings or get_settings()
    logger.info(
        "starting ETL flow (prefect={}) persist_to_db={}",
        PREFECT_AVAILABLE,
        persist_to_db,
    )

    extracted = extract_task(source, settings)
    transformed = transform_task(extracted, settings, default_stay_date)
    load_result = load_task(transformed, settings, persist_to_db)

    summary: dict[str, Any] = {
        "prefect": PREFECT_AVAILABLE,
        "extracted": extracted.row_counts(),
        "transform": transformed.meta,
        "load": {
            "feature_path": load_result.feature_path,
            "observations_path": load_result.observations_path,
            "db_loaded": load_result.db_loaded,
            "db_counts": load_result.db_counts,
            "warnings": load_result.warnings,
        },
        "quality": transformed.quality.to_dict() if transformed.quality else None,
    }
    logger.info("ETL flow complete: {}", summary["load"])
    return summary


def run_etl(
    *,
    source: DataSource | None = None,
    settings: Settings | None = None,
    default_stay_date: date | None = None,
    persist_to_db: bool = True,
) -> dict[str, Any]:
    """Convenience entrypoint for services/CLIs that does not require Prefect.

    Delegates to :func:`etl_flow`; when Prefect is installed this still executes
    the flow (synchronously), and when it is not, it runs the plain functions.
    """
    return etl_flow(
        source=source,
        settings=settings,
        default_stay_date=default_stay_date,
        persist_to_db=persist_to_db,
    )
