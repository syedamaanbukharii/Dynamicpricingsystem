"""Training service.

Thin orchestration around :class:`~app.training.train.ModelTrainer`. It loads a
training frame (from a supplied DataFrame or a CSV file), runs training, records
a Prometheus metric, and best-effort-persists a :class:`ModelRun` row describing
the run. Database persistence failures are non-fatal so training succeeds even
without a live database.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import Settings, get_settings
from app.monitoring import record_training
from app.training import ModelTrainer, TrainingResult
from app.utils.exceptions import TrainingError
from app.utils.logging import get_logger

logger = get_logger("ml")


def _persist_model_run(result: TrainingResult) -> None:
    """Best-effort insert of a model-run record into the database."""
    try:
        from app.database import ModelRun, get_session, init_db

        init_db()
        with get_session() as session:
            session.add(
                ModelRun(
                    version=result.version,
                    trained_at=datetime.fromisoformat(result.trained_at),
                    rmse=float(result.metrics.get("rmse", 0.0)),
                    mae=float(result.metrics.get("mae", 0.0)),
                    r2=float(result.metrics.get("r2", 0.0)),
                    mape=float(result.metrics.get("mape", 0.0)),
                    n_train=result.n_train,
                    n_val=result.n_val,
                    params=dict(result.best_params),
                    feature_importance=dict(result.feature_importance),
                    model_path=result.model_path,
                )
            )
        logger.info("recorded model run {} in database", result.version)
    except Exception as exc:
        logger.warning("model-run DB record skipped: {}", exc)


def train_from_frame(
    raw: pd.DataFrame,
    *,
    settings: Settings | None = None,
    val_fraction: float = 0.2,
    persist_run: bool = True,
) -> TrainingResult:
    """Train a model from an in-memory observation frame.

    Args:
        raw: Observation frame including the ``rooms_sold`` target column.
        settings: Application settings (defaults to the cached singleton).
        val_fraction: Fraction of (chronologically last) rows held out for
            validation.
        persist_run: Whether to record the run in the database (best-effort).

    Returns:
        The :class:`TrainingResult` summarizing the run.
    """
    settings = settings or get_settings()
    trainer = ModelTrainer(settings)
    try:
        result = trainer.train(raw, val_fraction=val_fraction)
    except Exception as exc:
        record_training("failure")
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError("Model training failed.", details={"error": str(exc)}) from exc

    record_training("success")
    logger.info(
        "training complete version={} rmse={:.3f} r2={:.3f}",
        result.version,
        result.metrics.get("rmse", float("nan")),
        result.metrics.get("r2", float("nan")),
    )
    if persist_run:
        _persist_model_run(result)
    return result


def train_from_file(
    path: Path | str,
    *,
    settings: Settings | None = None,
    val_fraction: float = 0.2,
    persist_run: bool = True,
) -> TrainingResult:
    """Train a model from a CSV/Parquet observation file.

    Args:
        path: Path to a CSV or Parquet file with the observation frame.
        settings: Application settings (defaults to the cached singleton).
        val_fraction: Validation hold-out fraction.
        persist_run: Whether to record the run in the database.

    Returns:
        The :class:`TrainingResult` summarizing the run.
    """
    path = Path(path)
    if not path.exists():
        raise TrainingError("Training data file not found.", details={"path": str(path)})
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, parse_dates=["stay_date", "booking_date"])
    logger.info("loaded {} training rows from {}", len(raw), path)
    return train_from_frame(
        raw, settings=settings, val_fraction=val_fraction, persist_run=persist_run
    )
