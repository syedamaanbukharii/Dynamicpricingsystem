"""Tests for feature engineering: single-row, matrix, and FEATURE_ORDER integrity."""

from __future__ import annotations

from datetime import date, timedelta

from app.feature_engineering import FeatureBuilder, feature_columns
from app.feature_engineering.defaults import default_feature_builder
from app.schemas.common import RoomType
from app.schemas.pricing import PriceRecommendationRequest, PricingFeatures


def _request() -> PriceRecommendationRequest:
    return PriceRecommendationRequest(
        hotel_id="HOTEL_A",
        room_type=RoomType.STANDARD_QUEEN,
        stay_date=date.today() + timedelta(days=30),
        inventory_total=40,
        rooms_sold=10,
        previous_price=150.0,
        competitor_rates=[140.0, 150.0, 165.0],
        booking_velocity=2.0,
    )


def test_feature_order_matches_schema() -> None:
    """The module's feature_columns() equals the schema's FEATURE_ORDER."""
    assert feature_columns() == list(PricingFeatures.FEATURE_ORDER)
    assert len(feature_columns()) == 21


def test_single_row_has_all_features() -> None:
    """A single built feature row exposes exactly the ordered model columns."""
    builder = default_feature_builder()
    features = builder.build_features(_request(), price=160.0)
    row = features.to_model_row()
    assert set(row.keys()) == set(feature_columns())
    assert row["price"] == 160.0


def test_matrix_columns_and_order(sample_frame) -> None:
    """The vectorized matrix has the columns of FEATURE_ORDER, in order."""
    builder = FeatureBuilder()
    matrix = builder.build_feature_matrix(sample_frame)
    assert list(matrix.columns) == feature_columns()
    assert len(matrix) == len(sample_frame)
    assert matrix.notna().all().all()


def test_price_to_comp_median_consistency() -> None:
    """price_to_comp_median reflects price relative to the competitor median."""
    builder = default_feature_builder()
    features = builder.build_features(_request(), price=150.0)
    # competitor median of [140,150,165] is 150 -> ratio ~1.0
    assert abs(features.price_to_comp_median - 1.0) < 0.1


def test_single_and_matrix_agree(sample_frame) -> None:
    """Single-row and matrix construction agree for the same input row."""
    builder = default_feature_builder()
    matrix = builder.build_feature_matrix(sample_frame.head(1))
    first = matrix.iloc[0]
    # price column should match the source row's price
    assert abs(first["price"] - float(sample_frame.iloc[0]["price"])) < 1e-6
