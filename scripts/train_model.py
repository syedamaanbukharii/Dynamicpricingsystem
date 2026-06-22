"""CLI: train the XGBoost demand model from an observation file.

Usage:
    python -m scripts.train_model [--data data/raw/observations.csv]
                                  [--val-fraction 0.2] [--no-db]

If ``--data`` is omitted and no observations file exists, sample data is
generated automatically so the command works out of the box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.services.training_service import train_from_file
from app.utils.logging import configure_logging, get_logger

from scripts.sample_data import generate_training_frame

logger = get_logger("ml")


def main() -> None:
    """Entry point for the training CLI."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the demand model.")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to an observation CSV/Parquet file."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="Validation hold-out fraction."
    )
    parser.add_argument(
        "--no-db", action="store_true", help="Do not record the run in the database."
    )
    args = parser.parse_args()

    settings = get_settings()
    data_path = Path(args.data) if args.data else Path(settings.raw_data_dir) / "observations.csv"

    if not data_path.exists():
        logger.info("no data at {}; generating sample observations", data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_training_frame().to_csv(data_path, index=False)

    result = train_from_file(
        data_path,
        settings=settings,
        val_fraction=args.val_fraction,
        persist_run=not args.no_db,
    )
    print(
        json.dumps(
            {
                "version": result.version,
                "model_path": result.model_path,
                "metrics": {k: round(v, 4) for k, v in result.metrics.items()},
                "n_train": result.n_train,
                "n_val": result.n_val,
                "top_features": dict(
                    sorted(
                        result.feature_importance.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:5]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
