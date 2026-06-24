"""ETL load stage.

Persists transformed artifacts to two destinations:

* the **feature store** and **processed** directories as columnar files
  (Parquet when a parquet engine is installed, otherwise CSV as a graceful
  fallback so the pipeline still completes in minimal environments); and
* **PostgreSQL**, via the SQLAlchemy ORM, for competitor rates, bookings,
  events, and holidays.

Database loading is best-effort: if the driver is missing or the database is
unreachable, the failure is logged and the file outputs still succeed, so the
pipeline remains runnable locally without a live database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.config import Settings, get_settings
from app.etl.transform import TransformedData
from app.utils.logging import get_logger

logger = get_logger("etl")


@dataclass
class LoadResult:
    """Summary of what the load stage wrote and where."""

    feature_path: str | None = None
    observations_path: str | None = None
    db_loaded: bool = False
    db_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    """Write a frame as Parquet, falling back to CSV if no engine is available.

    Returns:
        The path actually written (extension may change to ``.csv`` on
        fallback).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        logger.info("wrote {} rows -> {}", len(frame), path)
        return str(path)
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        logger.warning("parquet unavailable ({}); wrote CSV fallback -> {}", exc, csv_path)
        return str(csv_path)


def load_feature_store(
    features: pd.DataFrame,
    *,
    settings: Settings | None = None,
    name: str = "features.parquet",
) -> str | None:
    """Persist the engineered feature matrix to the feature store directory."""
    if features is None or features.empty:
        logger.warning("no features to write to feature store")
        return None
    settings = settings or get_settings()
    return _write_frame(features, Path(settings.feature_store_dir) / name)


def load_processed(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    name: str = "observations.parquet",
) -> str | None:
    """Persist a processed/cleaned frame to the processed-data directory."""
    if frame is None or frame.empty:
        return None
    settings = settings or get_settings()
    return _write_frame(frame, Path(settings.processed_data_dir) / name)


def _load_database(transformed: TransformedData) -> dict[str, int]:
    """Insert competitor rates, bookings, events, and holidays via the ORM.

    Returns a per-table inserted-row count. Raises on any failure so the caller
    can decide whether to treat DB loading as fatal.
    """
    from app.database import (
        Booking,
        CompetitorRate,
        Event,
        Holiday,
        get_session,
        init_db,
    )

    init_db()
    counts: dict[str, int] = {
        "competitor_rates": 0,
        "bookings": 0,
        "events": 0,
        "holidays": 0,
    }

    with get_session() as session:
        for rate in transformed.competitor_rates:
            session.add(
                CompetitorRate(
                    source=rate.source,
                    competitor=rate.competitor,
                    room_type=rate.room_type.value,
                    raw_room_name=rate.raw_room_name,
                    price=float(rate.price),
                    currency=rate.currency.value,
                    stay_date=rate.stay_date,
                    scraped_at=rate.scraped_at,
                )
            )
            counts["competitor_rates"] += 1

        if not transformed.bookings.empty:
            required = {"hotel_id", "booking_id", "room_type", "booking_date", "stay_date", "price"}
            if required.issubset(transformed.bookings.columns):
                for row in transformed.bookings.to_dict(orient="records"):
                    session.add(
                        Booking(
                            hotel_id=str(row["hotel_id"]),
                            booking_id=str(row["booking_id"]),
                            room_type=str(row["room_type"]),
                            booking_date=row["booking_date"],
                            stay_date=row["stay_date"],
                            nights=int(row.get("nights", 1) or 1),
                            rooms=int(row.get("rooms", 1) or 1),
                            price=float(row["price"]),
                            currency=str(row.get("currency", "USD") or "USD"),
                            channel=str(row.get("channel", "DIRECT") or "DIRECT"),
                            is_cancelled=bool(row.get("is_cancelled", False)),
                        )
                    )
                    counts["bookings"] += 1

        if not transformed.events.empty and {"name", "start_date", "end_date"}.issubset(
            transformed.events.columns
        ):
            for row in transformed.events.to_dict(orient="records"):
                session.add(
                    Event(
                        name=str(row["name"]),
                        start_date=row["start_date"],
                        end_date=row["end_date"],
                        venue=(str(row["venue"]) if row.get("venue") else None),
                        expected_attendance=int(row.get("expected_attendance", 0) or 0),
                        demand_multiplier=float(row.get("demand_multiplier", 1.0) or 1.0),
                    )
                )
                counts["events"] += 1

        if not transformed.holidays.empty and {"name", "holiday_date"}.issubset(
            transformed.holidays.columns
        ):
            for row in transformed.holidays.to_dict(orient="records"):
                session.add(
                    Holiday(
                        name=str(row["name"]),
                        holiday_date=row["holiday_date"],
                        country=str(row.get("country", "US") or "US"),
                        region=(str(row["region"]) if row.get("region") else None),
                        is_public=bool(row.get("is_public", True)),
                    )
                )
                counts["holidays"] += 1

    return counts


def load(
    transformed: TransformedData,
    *,
    settings: Settings | None = None,
    persist_to_db: bool = True,
) -> LoadResult:
    """Run the load stage.

    Args:
        transformed: Output of :func:`app.etl.transform.transform`.
        settings: Application settings (defaults to the cached singleton).
        persist_to_db: Whether to attempt a PostgreSQL load. When the database
            is unavailable, the error is captured as a warning and file outputs
            still succeed.

    Returns:
        A :class:`LoadResult` describing what was written.
    """
    settings = settings or get_settings()
    result = LoadResult()

    result.feature_path = load_feature_store(transformed.features, settings=settings)
    result.observations_path = load_processed(transformed.observations, settings=settings)

    if persist_to_db:
        try:
            result.db_counts = _load_database(transformed)
            result.db_loaded = True
            logger.info("database load complete: {}", result.db_counts)
        except Exception as exc:
            msg = f"database load skipped: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)

    return result
