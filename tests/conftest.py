"""Shared pytest fixtures and test configuration.

Sets a deterministic, offline-friendly environment (LLM disabled, human logs,
tiny Optuna budget) before importing application modules, adds the repo root to
``sys.path``, and exposes reusable fixtures: settings, a sample observation
frame, a trained-model directory, and a FastAPI ``TestClient``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure a deterministic, dependency-light test environment *before* app imports.
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("OPTUNA_TRIALS", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def settings():
    """Return the application settings singleton for the test environment."""
    from app.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """A small, deterministic observation frame suitable for training/features."""
    from scripts.sample_data import generate_training_frame

    return generate_training_frame(days=120, seed=7)


@pytest.fixture
def raw_listings():
    """Raw, messy competitor listings parsed into schema models."""
    from app.schemas.competitor import RawCompetitorListing
    from scripts.sample_data import generate_raw_competitor_listings

    return [RawCompetitorListing.model_validate(d) for d in generate_raw_competitor_listings()]


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, sample_frame) -> Path:
    """Train a real (tiny) model into a temp dir and return its path."""
    from app.config import get_settings
    from app.training import ModelTrainer

    model_dir = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(model_dir)
    get_settings.cache_clear()
    trainer = ModelTrainer(get_settings())
    trainer.train(sample_frame, val_fraction=0.2)
    return model_dir


@pytest.fixture
def client():
    """A FastAPI TestClient against a freshly built app (heuristic model)."""
    from app.api.main import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
