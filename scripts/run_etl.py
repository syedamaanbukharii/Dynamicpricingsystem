"""CLI: run the ETL pipeline over ``data/raw``.

Usage:
    python -m scripts.run_etl [--no-db] [--default-stay 2026-07-15]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from app.services.etl_service import run_etl_pipeline
from app.utils.logging import configure_logging


def main() -> None:
    """Entry point for the ETL CLI."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the ETL pipeline.")
    parser.add_argument("--no-db", action="store_true", help="Skip the PostgreSQL load step.")
    parser.add_argument(
        "--default-stay",
        default=None,
        help="Stay date (YYYY-MM-DD) for competitor listings missing one.",
    )
    args = parser.parse_args()

    default_stay = (
        datetime.strptime(args.default_stay, "%Y-%m-%d").date() if args.default_stay else None
    )
    summary = run_etl_pipeline(default_stay_date=default_stay, persist_to_db=not args.no_db)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
