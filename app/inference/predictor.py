"""Load a trained XGBoost model and serve demand predictions.

:class:`DemandPredictor` implements the same :class:`DemandModel` interface as
the heuristic fallback, so the pricing engine is agnostic to which is in use. It
validates that the loaded model's feature order matches the current code to
prevent silent train/serve skew.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from app.config import Settings, get_settings
from app.feature_engineering.features import feature_columns
from app.inference.base import DemandModel
from app.schemas.pricing import PricingFeatures
from app.utils.exceptions import ModelNotFoundError
from app.utils.logging import get_logger

logger = get_logger("ml")


def _resolve_version_dir(model_dir: Path) -> Path:
    """Return the directory of the latest trained model, if any."""
    latest_pointer = model_dir / "latest.txt"
    if latest_pointer.exists():
        candidate = model_dir / latest_pointer.read_text().strip()
        if (candidate / "model.joblib").exists():
            return candidate
    candidates = sorted(
        (p for p in model_dir.glob("model_*") if (p / "model.joblib").exists()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise ModelNotFoundError(
            "No trained model found. Train one via the training pipeline first.",
            details={"model_dir": str(model_dir)},
        )
    return candidates[-1]


class DemandPredictor(DemandModel):
    """XGBoost-backed demand model loaded from disk."""

    def __init__(self, model: object, metadata: dict[str, object]) -> None:
        self._model = model
        self._metadata = metadata
        self._version = str(metadata.get("version", "unknown"))
        loaded_features = list(metadata.get("feature_names", []))
        expected = feature_columns()
        if loaded_features and loaded_features != expected:
            raise ModelNotFoundError(
                "Model feature schema does not match current code (train/serve skew).",
                details={"expected": expected, "loaded": loaded_features},
            )

    @property
    def version(self) -> str:  # noqa: D102
        return self._version

    @property
    def metadata(self) -> dict[str, object]:
        """Return the model's persisted metadata (metrics, params, importance)."""
        return dict(self._metadata)

    @classmethod
    def load(cls, settings: Settings | None = None) -> DemandPredictor:
        """Load the latest model from the configured model directory."""
        settings = settings or get_settings()
        version_dir = _resolve_version_dir(settings.model_dir)
        model = joblib.load(version_dir / "model.joblib")
        metadata = json.loads((version_dir / "metadata.json").read_text())
        logger.info("loaded model version={}", metadata.get("version"))
        return cls(model, metadata)

    def _frame(self, rows: list[PricingFeatures]) -> pd.DataFrame:
        """Assemble a feature DataFrame in the canonical column order."""
        return pd.DataFrame([r.to_model_row() for r in rows])[feature_columns()]

    def predict_rooms_sold(self, features: PricingFeatures) -> float:  # noqa: D102
        return self.predict_many([features])[0]

    def predict_many(self, rows: list[PricingFeatures]) -> list[float]:  # noqa: D102
        frame = self._frame(rows)
        preds = self._model.predict(frame.to_numpy())
        return [max(0.0, float(p)) for p in preds]


def load_demand_model(settings: Settings | None = None) -> DemandModel:
    """Return the trained model, or the heuristic fallback if none is available."""
    from app.inference.heuristic import HeuristicDemandModel

    settings = settings or get_settings()
    try:
        return DemandPredictor.load(settings)
    except ModelNotFoundError:
        logger.warning("no trained model available; using heuristic demand model")
        return HeuristicDemandModel()
