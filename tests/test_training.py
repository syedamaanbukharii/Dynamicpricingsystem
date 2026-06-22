"""Integration tests for the real training + inference path (XGBoost)."""

from __future__ import annotations

import os
from datetime import date, timedelta

from app.config import get_settings
from app.feature_engineering import feature_columns
from app.feature_engineering.defaults import default_feature_builder
from app.inference.predictor import DemandPredictor, load_demand_model
from app.schemas.common import RoomType
from app.schemas.pricing import PriceRecommendationRequest
from app.training import ModelTrainer
from app.training.dataset import prepare_dataset
from app.training.evaluate import regression_metrics


def test_prepare_dataset_chronological_split(sample_frame) -> None:
    """Dataset preparation splits chronologically with aligned feature columns."""
    prepared = prepare_dataset(sample_frame, val_fraction=0.2)
    assert list(prepared.x_train.columns) == feature_columns()
    assert len(prepared.x_train) > len(prepared.x_val)
    assert len(prepared.y_train) == len(prepared.x_train)


def test_regression_metrics_perfect_prediction() -> None:
    """Metrics on a perfect prediction are near-zero error and r2 == 1."""
    import numpy as np

    y = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_train_persists_and_loads(tmp_path, sample_frame) -> None:
    """A trained model persists to disk, reloads, and matches the feature schema."""
    os.environ["MODEL_DIR"] = str(tmp_path)
    get_settings.cache_clear()
    settings = get_settings()

    result = ModelTrainer(settings).train(sample_frame, val_fraction=0.2)
    assert result.metrics["r2"] > 0.5  # learns meaningful signal on sample data
    assert (tmp_path / "latest.txt").exists()

    model = load_demand_model(settings)
    assert isinstance(model, DemandPredictor)
    assert model.version == result.version

    # Reset global settings for subsequent tests.
    os.environ.pop("MODEL_DIR", None)
    get_settings.cache_clear()


def test_trained_model_demand_decreases_with_price(tmp_path, sample_frame) -> None:
    """The trained model exhibits a downward-sloping demand curve in price."""
    os.environ["MODEL_DIR"] = str(tmp_path)
    get_settings.cache_clear()
    settings = get_settings()
    ModelTrainer(settings).train(sample_frame, val_fraction=0.2)
    model = load_demand_model(settings)

    builder = default_feature_builder()
    request = PriceRecommendationRequest(
        hotel_id="HOTEL_A",
        room_type=RoomType.DELUXE_KING,
        stay_date=date.today() + timedelta(days=21),
        inventory_total=50,
        rooms_sold=20,
        previous_price=180.0,
        competitor_rates=[180.0, 195.0, 210.0],
        booking_velocity=3.0,
    )
    low = model.predict_rooms_sold(builder.build_features(request, 140.0))
    high = model.predict_rooms_sold(builder.build_features(request, 320.0))
    assert high <= low + 1e-6

    os.environ.pop("MODEL_DIR", None)
    get_settings.cache_clear()
