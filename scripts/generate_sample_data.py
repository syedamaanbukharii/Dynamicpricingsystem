"""CLI: generate a realistic sample data drop into ``data/raw``.

Writes the canonical raw inputs the ETL pipeline consumes so the project is
runnable end-to-end immediately after cloning:

* ``observations.csv``        - the training observation frame (with target);
* ``competitor_listings.json``- messy, scraper-like competitor listings;
* ``events.csv`` / ``holidays.csv`` - calendar inputs;
* ``inventory.csv``           - derived per (hotel, room, stay) inventory.

Usage:
    python -m scripts.generate_sample_data [--days 540] [--out data/raw]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from app.config import get_settings
from app.feature_engineering.defaults import default_events, default_holidays
from app.utils.logging import configure_logging, get_logger

from scripts.sample_data import (
    generate_raw_competitor_listings,
    generate_training_frame,
)

logger = get_logger("etl")


def _write_events(path: Path) -> int:
    rows = [
        {
            "name": e.name,
            "start_date": e.start_date.isoformat(),
            "end_date": e.end_date.isoformat(),
            "venue": e.venue or "",
            "expected_attendance": e.expected_attendance,
            "demand_multiplier": e.demand_multiplier,
        }
        for e in default_events()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def _write_holidays(path: Path) -> int:
    rows = [
        {
            "name": h.name,
            "holiday_date": h.holiday_date.isoformat(),
            "country": h.country,
            "region": h.region or "",
            "is_public": h.is_public,
        }
        for h in default_holidays()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def _write_inventory(observations: pd.DataFrame, path: Path) -> int:
    inv = (
        observations[["hotel_id", "room_type", "stay_date", "inventory_total"]]
        .drop_duplicates(subset=["hotel_id", "room_type", "stay_date"])
        .reset_index(drop=True)
    )
    inv.to_csv(path, index=False)
    return len(inv)


def main() -> None:
    """Entry point for the sample-data generator."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate sample raw data.")
    parser.add_argument("--days", type=int, default=540, help="History length in days.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: settings.raw_data_dir).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.out) if args.out else Path(settings.raw_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = generate_training_frame(days=args.days, seed=args.seed)
    obs_path = out_dir / "observations.csv"
    observations.to_csv(obs_path, index=False)
    logger.info("wrote {} observation rows -> {}", len(observations), obs_path)

    listings = generate_raw_competitor_listings()
    comp_path = out_dir / "competitor_listings.json"
    comp_path.write_text(json.dumps(listings, indent=2, default=str), encoding="utf-8")
    logger.info("wrote {} competitor listings -> {}", len(listings), comp_path)

    n_events = _write_events(out_dir / "events.csv")
    n_holidays = _write_holidays(out_dir / "holidays.csv")
    n_inv = _write_inventory(observations, out_dir / "inventory.csv")
    logger.info(
        "wrote events={} holidays={} inventory={} into {}",
        n_events,
        n_holidays,
        n_inv,
        out_dir,
    )
    print(
        f"Sample data written to {out_dir} "
        f"(observations={len(observations)}, listings={len(listings)}, "
        f"events={n_events}, holidays={n_holidays}, inventory={n_inv})."
    )


if __name__ == "__main__":
    main()
