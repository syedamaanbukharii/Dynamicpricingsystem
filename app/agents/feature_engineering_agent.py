"""Feature Engineering Agent.

Orchestrates the deterministic :class:`~app.feature_engineering.features.FeatureBuilder`
into a single, observable step the rest of the platform (ETL, training, API) can
call. It validates inputs, runs a lightweight quality gate, builds the model
feature matrix, and reports which engineered columns were produced. Feature
construction itself is intentionally deterministic (no LLM) so training and
serving stay perfectly aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.feature_engineering.defaults import default_feature_builder
from app.feature_engineering.features import FeatureBuilder, feature_columns
from app.utils.exceptions import DataQualityError
from app.utils.logging import get_logger

logger = get_logger("features")

_REQUIRED_INPUT_COLUMNS = ("stay_date", "price", "inventory_total", "rooms_sold")


@dataclass
class FeatureEngineeringResult:
    """A built feature matrix plus metadata about the engineering run."""

    features: pd.DataFrame
    feature_names: list[str]
    n_rows: int
    dropped_rows: int = 0
    warnings: list[str] = field(default_factory=list)


class FeatureEngineeringAgent:
    """Builds engineered features from a raw observation frame."""

    def __init__(self, builder: FeatureBuilder | None = None) -> None:
        self._builder = builder or default_feature_builder()

    def _validate(self, df: pd.DataFrame) -> None:
        """Ensure the minimum input columns are present."""
        missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise DataQualityError(
                "Input frame is missing required columns for feature engineering.",
                details={"missing": missing},
            )

    def build(self, df: pd.DataFrame) -> FeatureEngineeringResult:
        """Validate, clean, and build the engineered feature matrix."""
        if df is None or df.empty:
            raise DataQualityError("No rows supplied for feature engineering.")
        self._validate(df)

        warnings: list[str] = []
        cleaned = df.copy()
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=list(_REQUIRED_INPUT_COLUMNS))
        dropped = before - len(cleaned)
        if dropped:
            warnings.append(f"Dropped {dropped} rows with missing required fields.")

        invalid_price = (cleaned["price"] <= 0).sum()
        if invalid_price:
            warnings.append(f"{int(invalid_price)} rows have non-positive prices.")
            cleaned = cleaned[cleaned["price"] > 0]

        features = self._builder.build_feature_matrix(cleaned)
        logger.info(
            "feature engineering agent produced {} rows x {} features",
            len(features),
            len(feature_columns()),
        )
        return FeatureEngineeringResult(
            features=features,
            feature_names=feature_columns(),
            n_rows=len(features),
            dropped_rows=dropped,
            warnings=warnings,
        )
