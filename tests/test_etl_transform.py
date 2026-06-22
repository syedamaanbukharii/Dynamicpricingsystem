"""Tests for the ETL transform stage and the file-based pipeline."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from app.etl.extract import FileDataSource, extract
from app.etl.transform import transform
from app.feature_engineering import feature_columns
from scripts.sample_data import (
    generate_raw_competitor_listings,
    generate_training_frame,
)


def _make_raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    generate_training_frame(days=90, seed=11).to_csv(raw / "observations.csv", index=False)
    (raw / "competitor_listings.json").write_text(
        json.dumps(generate_raw_competitor_listings(), default=str), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "name": "Tech Conf",
                "start_date": "2026-06-15",
                "end_date": "2026-06-18",
                "venue": "Center",
                "expected_attendance": 12000,
                "demand_multiplier": 1.6,
            }
        ]
    ).to_csv(raw / "events.csv", index=False)
    return raw


def test_transform_cleans_competitors_and_builds_features(tmp_path: Path) -> None:
    """Transform yields cleaned rates and a full engineered feature matrix."""
    raw = _make_raw_dir(tmp_path)
    extracted = extract(FileDataSource(raw))
    result = transform(extracted, default_stay_date=date.today() + timedelta(days=5))

    assert len(result.competitor_rates) >= 1
    assert result.quality is not None and result.quality.is_ok
    assert not result.features.empty
    assert list(result.features.columns) == feature_columns()
    assert len(result.features) == len(result.observations)


def test_transform_tolerates_missing_inputs(tmp_path: Path) -> None:
    """Missing optional inputs (events/holidays/bookings) do not break transform."""
    raw = tmp_path / "raw"
    raw.mkdir()
    generate_training_frame(days=60, seed=3).to_csv(raw / "observations.csv", index=False)
    extracted = extract(FileDataSource(raw))
    result = transform(extracted)
    assert not result.observations.empty
    assert result.bookings.empty  # none supplied
    assert len(result.competitor_rates) == 0  # none supplied


def test_transform_canonicalizes_room_types(tmp_path: Path) -> None:
    """Free-text room types in observations are normalized to the enum values."""
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = generate_training_frame(days=60, seed=5)
    frame["room_type"] = "Deluxe King Room"  # free text variant
    frame.to_csv(raw / "observations.csv", index=False)
    extracted = extract(FileDataSource(raw))
    result = transform(extracted)
    assert set(result.observations["room_type"].unique()) <= {"DELUXE_KING"}
