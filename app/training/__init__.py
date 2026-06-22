"""Training package: dataset prep, evaluation, and the XGBoost trainer."""

from app.training.dataset import TARGET_COLUMN, PreparedData, prepare_dataset
from app.training.evaluate import regression_metrics
from app.training.train import ModelTrainer, TrainingResult

__all__ = [
    "TARGET_COLUMN",
    "ModelTrainer",
    "PreparedData",
    "TrainingResult",
    "prepare_dataset",
    "regression_metrics",
]
