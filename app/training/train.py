"""XGBoost demand-model training with Optuna tuning and MLflow tracking.

The trainer fits an :class:`xgboost.XGBRegressor` that predicts rooms sold from
the engineered features (including price), enabling the pricing engine to find a
revenue-maximizing rate. Hyperparameters are tuned with Optuna using
time-series cross-validation; the best model is persisted with full metadata and
its feature-importance report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from app.config import Settings, get_settings
from app.training.dataset import PreparedData, prepare_dataset
from app.training.evaluate import regression_metrics
from app.utils.exceptions import TrainingError
from app.utils.logging import get_logger

logger = get_logger("ml")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_STATIC_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
}


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    version: str
    model_path: str
    metrics: dict[str, float]
    best_params: dict[str, float]
    feature_importance: dict[str, float]
    n_train: int
    n_val: int
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _suggest_params(trial: optuna.Trial, seed: int) -> dict[str, object]:
    """Optuna search space for the XGBoost regressor."""
    return {
        **_STATIC_PARAMS,
        "random_state": seed,
        "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


class ModelTrainer:
    """Trains, evaluates, and persists the XGBoost demand model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _cv_rmse(self, params: dict[str, object], data: PreparedData) -> float:
        """Mean RMSE across time-series CV folds on the training partition."""
        n_splits = max(2, min(self.settings.cv_splits, len(data.x_train) // 5))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        scores: list[float] = []
        x = data.x_train.to_numpy()
        y = data.y_train.to_numpy()
        for train_idx, val_idx in splitter.split(x):
            model = XGBRegressor(**params)
            model.fit(x[train_idx], y[train_idx])
            preds = model.predict(x[val_idx])
            scores.append(float(np.sqrt(np.mean((preds - y[val_idx]) ** 2))))
        return float(np.mean(scores))

    def _tune(self, data: PreparedData) -> dict[str, object]:
        """Run Optuna tuning and return the best hyperparameters."""
        trials = self.settings.optuna_trials
        if trials <= 0:
            logger.info("optuna disabled; using default hyperparameters")
            return {
                **_STATIC_PARAMS,
                "random_state": self.settings.random_seed,
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            }

        sampler = optuna.samplers.TPESampler(seed=self.settings.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda t: self._cv_rmse(_suggest_params(t, self.settings.random_seed), data),
            n_trials=trials,
            show_progress_bar=False,
        )
        logger.info("optuna best CV RMSE={:.4f}", study.best_value)
        return {
            **_STATIC_PARAMS,
            "random_state": self.settings.random_seed,
            **study.best_params,
        }

    def train(self, raw: pd.DataFrame, *, val_fraction: float = 0.2) -> TrainingResult:
        """Train, evaluate, persist, and (best-effort) MLflow-log a model."""
        if raw is None or raw.empty:
            raise TrainingError("no training data provided")

        data = prepare_dataset(raw, val_fraction=val_fraction)
        best_params = self._tune(data)

        model = XGBRegressor(**best_params)
        model.fit(data.x_train.to_numpy(), data.y_train.to_numpy())

        val_pred = model.predict(data.x_val.to_numpy())
        metrics = regression_metrics(data.y_val.to_numpy(), val_pred)
        logger.info(
            "validation metrics: RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}",
            **metrics,
        )

        importance = dict(
            sorted(
                zip(data.feature_names, model.feature_importances_.tolist(), strict=True),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

        version = f"model_{datetime.now(UTC):%Y%m%d_%H%M%S}"
        model_path = self._persist(version, model, data, metrics, best_params, importance)
        result = TrainingResult(
            version=version,
            model_path=str(model_path),
            metrics=metrics,
            best_params={k: v for k, v in best_params.items() if isinstance(v, (int, float))},
            feature_importance=importance,
            n_train=len(data.x_train),
            n_val=len(data.x_val),
        )
        self._log_mlflow(result)
        return result

    def _persist(
        self,
        version: str,
        model: XGBRegressor,
        data: PreparedData,
        metrics: dict[str, float],
        params: dict[str, object],
        importance: dict[str, float],
    ) -> Path:
        """Write the model and metadata under ``model_dir/<version>/``."""
        out_dir = self.settings.model_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)

        metadata = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "feature_names": data.feature_names,
            "metrics": metrics,
            "params": {k: v for k, v in params.items() if not callable(v)},
            "feature_importance": importance,
            "n_train": len(data.x_train),
            "n_val": len(data.x_val),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        (self.settings.model_dir / "latest.txt").write_text(version)
        logger.info("persisted model -> {}", model_path)
        return model_path

    def _log_mlflow(self, result: TrainingResult) -> None:
        """Best-effort MLflow logging; a tracking outage must not fail training."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment)
            with mlflow.start_run(run_name=result.version):
                mlflow.log_params(result.best_params)
                mlflow.log_metrics(result.metrics)
                mlflow.log_dict(result.feature_importance, "feature_importance.json")
        except Exception as exc:
            logger.warning("MLflow logging skipped: {}", exc)
