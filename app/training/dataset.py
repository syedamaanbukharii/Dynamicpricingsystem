"""Dataset preparation with time-aware splitting for leakage-free evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.feature_engineering.features import FeatureBuilder, feature_columns
from app.utils.exceptions import TrainingError
from app.utils.logging import get_logger

logger = get_logger("ml")

TARGET_COLUMN = "rooms_sold"


@dataclass
class PreparedData:
    """Feature matrix / target split chronologically into train and validation."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    feature_names: list[str]


def prepare_dataset(
    raw: pd.DataFrame,
    feature_builder: FeatureBuilder | None = None,
    *,
    val_fraction: float = 0.2,
    date_column: str = "stay_date",
) -> PreparedData:
    """Build features and split chronologically (oldest -> train, newest -> val).

    A time-ordered split avoids look-ahead leakage that random splits would
    introduce for a forecasting task.
    """
    if TARGET_COLUMN not in raw.columns:
        raise TrainingError(f"training data missing target column '{TARGET_COLUMN}'")
    if not 0.0 < val_fraction < 0.9:
        raise TrainingError("val_fraction must be in (0, 0.9)")
    if len(raw) < 20:
        raise TrainingError("need at least 20 rows to train a meaningful model")

    builder = feature_builder or FeatureBuilder()
    ordered = raw.sort_values(date_column).reset_index(drop=True)
    features = builder.build_feature_matrix(ordered)
    target = ordered[TARGET_COLUMN].astype(float).reset_index(drop=True)

    split_at = int(len(ordered) * (1.0 - val_fraction))
    x_train = features.iloc[:split_at].reset_index(drop=True)
    x_val = features.iloc[split_at:].reset_index(drop=True)
    y_train = target.iloc[:split_at].reset_index(drop=True)
    y_val = target.iloc[split_at:].reset_index(drop=True)

    logger.info(
        "prepared dataset: train={} val={} features={}",
        len(x_train),
        len(x_val),
        len(feature_columns()),
    )
    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_names=feature_columns(),
    )
