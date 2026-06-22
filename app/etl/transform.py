"""ETL transformation stage.

Turns raw extracted inputs into clean, validated, model-ready artifacts. The
stage is deliberately composed from the same agents the rest of the platform
uses, so the cleaning and feature logic stay identical between ETL and serving:

* competitor listings are cleaned and structured by the
  :class:`~app.agents.competitor_cleaning.CompetitorCleaningAgent` and graded by
  the :class:`~app.agents.data_quality.DataQualityAgent`;
* room-name strings on bookings/observations are canonicalized through the
  :class:`~app.agents.room_matching.RoomMatcher`;
* the engineered feature matrix is produced by the
  :class:`~app.agents.feature_engineering_agent.FeatureEngineeringAgent`, which
  wraps the very same deterministic ``FeatureBuilder`` used at inference time.

No LLM is required: when ``llm_enabled`` is false (or the SDK/key is absent) the
agents fall back to their deterministic rule-based paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.agents.competitor_cleaning import CompetitorCleaningAgent
from app.agents.data_quality import DataQualityAgent, QualityReport
from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.llm import get_llm_client
from app.agents.room_matching import RoomMatcher
from app.config import Settings, get_settings
from app.etl.extract import ExtractedData
from app.schemas.common import RoomType
from app.schemas.competitor import CompetitorRate, RawCompetitorListing
from app.training.dataset import TARGET_COLUMN
from app.utils.exceptions import ETLError
from app.utils.logging import get_logger

logger = get_logger("etl")

# Minimum columns an observation frame must carry to be model-ready.
_OBSERVATION_REQUIRED = ("hotel_id", "room_type", "stay_date", "price", "inventory_total")


@dataclass
class TransformedData:
    """Clean, validated outputs ready to be loaded.

    Attributes:
        observations: The cleaned training observation frame (one row per
            hotel/room/stay observation), including the ``rooms_sold`` target and
            all raw feature inputs. This is what the trainer consumes.
        features: The engineered feature matrix (``FEATURE_ORDER`` columns,
            all floats) aligned row-for-row with ``observations``. Persisted to
            the feature store.
        competitor_rates: Cleaned, structured competitor rates.
        bookings: Normalized bookings frame (canonical room types, deduped).
        events: Normalized local-events frame.
        holidays: Normalized holiday frame.
        quality: The competitor-rate quality report.
        meta: Free-form run metadata (counts, anomaly tallies, warnings).
    """

    observations: pd.DataFrame
    features: pd.DataFrame
    competitor_rates: list[CompetitorRate]
    bookings: pd.DataFrame
    events: pd.DataFrame
    holidays: pd.DataFrame
    quality: QualityReport | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _to_date_series(series: pd.Series) -> pd.Series:
    """Coerce a column to python ``date`` objects, leaving NaT as None."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def _canonicalize_room_types(
    frame: pd.DataFrame, matcher: RoomMatcher, *, column: str = "room_type"
) -> pd.DataFrame:
    """Map free-text room descriptions to canonical :class:`RoomType` values."""
    if frame.empty or column not in frame.columns:
        return frame
    out = frame.copy()
    unique_names = [str(v) for v in out[column].dropna().unique()]
    mapping = matcher.match_many(unique_names)

    def _map(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return RoomType.OTHER.value
        matched = mapping.get(str(value))
        if matched is not None:
            return matched.value
        # Already-canonical values pass through.
        try:
            return RoomType(str(value)).value
        except ValueError:
            return RoomType.OTHER.value

    out[column] = out[column].map(_map)
    return out


def _listings_to_models(
    listings: list[dict[str, Any]],
) -> list[RawCompetitorListing]:
    """Coerce raw listing dicts into validated :class:`RawCompetitorListing`."""
    models: list[RawCompetitorListing] = []
    for raw in listings:
        try:
            models.append(RawCompetitorListing.model_validate(raw))
        except Exception as exc:
            logger.warning("dropping malformed competitor listing: {}", exc)
    return models


def _clip_price_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Flag and drop non-positive prices; return cleaned frame and drop count.

    Extreme statistical outliers are *flagged* (logged) rather than removed, to
    avoid silently discarding genuine peak-demand pricing. Only invalid
    (non-positive / non-finite) prices are dropped.
    """
    if frame.empty or "price" not in frame.columns:
        return frame, 0
    out = frame.copy()
    price = pd.to_numeric(out["price"], errors="coerce")
    invalid_mask = ~np.isfinite(price) | (price <= 0)
    dropped = int(invalid_mask.sum())
    out = out.loc[~invalid_mask].copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    # Flag (do not drop) IQR outliers per room type for observability.
    if "room_type" in out.columns and not out.empty:
        flagged = 0
        for _, grp in out.groupby("room_type"):
            if len(grp) < 8:
                continue
            q1, q3 = grp["price"].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            hi = q3 + 3.0 * iqr
            flagged += int((grp["price"] > hi).sum())
        if flagged:
            logger.info("flagged {} statistical price outliers (kept)", flagged)
    return out, dropped


def _normalize_observations(
    frame: pd.DataFrame, matcher: RoomMatcher
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, canonicalize, dedupe, and clean the observation frame."""
    meta: dict[str, Any] = {}
    if frame.empty:
        return frame, {"reason": "no observations supplied"}

    missing = [c for c in _OBSERVATION_REQUIRED if c not in frame.columns]
    if missing:
        raise ETLError(
            "Observation frame missing required columns.",
            details={"missing": missing},
        )

    out = frame.copy()
    out["stay_date"] = _to_date_series(out["stay_date"])
    if "booking_date" in out.columns:
        out["booking_date"] = _to_date_series(out["booking_date"])
    out = out.dropna(subset=["stay_date"])

    out = _canonicalize_room_types(out, matcher)

    before = len(out)
    subset = [c for c in ("hotel_id", "room_type", "stay_date", "booking_date") if c in out.columns]
    out = out.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    meta["deduped_rows"] = before - len(out)

    out, invalid_prices = _clip_price_outliers(out)
    meta["dropped_invalid_price"] = invalid_prices

    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce").fillna(0)
    meta["rows"] = len(out)
    return out, meta


def transform(
    extracted: ExtractedData,
    *,
    settings: Settings | None = None,
    default_stay_date: date | None = None,
) -> TransformedData:
    """Run the transformation stage over extracted inputs.

    Args:
        extracted: The raw bundle from :func:`app.etl.extract.extract`.
        settings: Application settings (defaults to the cached singleton).
        default_stay_date: Stay date assigned to competitor listings that lack
            one (otherwise such listings are dropped by the cleaning agent).

    Returns:
        A :class:`TransformedData` bundle of cleaned, model-ready artifacts.
    """
    settings = settings or get_settings()
    llm = get_llm_client(settings)
    matcher = RoomMatcher(llm=llm)
    cleaner = CompetitorCleaningAgent(room_matcher=matcher, llm=llm)
    quality_agent = DataQualityAgent()

    meta: dict[str, Any] = {}

    # 1) Competitor listings -> structured, graded rates.
    listing_models = _listings_to_models(extracted.competitor_listings)
    cleaning = cleaner.clean(listing_models, default_stay_date=default_stay_date)
    quality = quality_agent.assess(cleaning.rates)
    meta["competitor"] = {
        "kept": cleaning.kept,
        "dropped": cleaning.removed,
        "quality_ok": quality.is_ok,
    }
    logger.info(
        "competitor transform: kept={} dropped={} quality_ok={}",
        cleaning.kept,
        cleaning.removed,
        quality.is_ok,
    )

    # 2) Bookings / events / holidays -> normalized frames for persistence.
    bookings = extracted.bookings
    if not bookings.empty:
        bookings = _canonicalize_room_types(bookings, matcher)
        for col in ("booking_date", "stay_date"):
            if col in bookings.columns:
                bookings[col] = _to_date_series(bookings[col])
        if {"hotel_id", "booking_id"}.issubset(bookings.columns):
            bookings = bookings.drop_duplicates(
                subset=["hotel_id", "booking_id"], keep="last"
            ).reset_index(drop=True)

    events = extracted.events
    if not events.empty:
        for col in ("start_date", "end_date"):
            if col in events.columns:
                events[col] = _to_date_series(events[col])

    holidays = extracted.holidays
    if not holidays.empty and "holiday_date" in holidays.columns:
        holidays["holiday_date"] = _to_date_series(holidays["holiday_date"])
        holidays = holidays.drop_duplicates(
            subset=[c for c in ("holiday_date", "country") if c in holidays.columns]
        ).reset_index(drop=True)

    # 3) Observation frame -> cleaned training frame + engineered features.
    observations, obs_meta = _normalize_observations(extracted.observations, matcher)
    meta["observations"] = obs_meta

    features = pd.DataFrame()
    if not observations.empty:
        try:
            fe_result = FeatureEngineeringAgent().build(observations)
            features = fe_result.features
            meta["features"] = {
                "rows": fe_result.n_rows,
                "dropped": fe_result.dropped_rows,
                "warnings": fe_result.warnings,
            }
        except Exception as exc:
            raise ETLError(
                "Feature engineering failed during transform.",
                details={"error": str(exc)},
            ) from exc

    return TransformedData(
        observations=observations,
        features=features,
        competitor_rates=cleaning.rates,
        bookings=bookings,
        events=events,
        holidays=holidays,
        quality=quality,
        meta=meta,
    )
