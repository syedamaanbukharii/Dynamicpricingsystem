"""ETL extraction stage.

Reads the seven raw inputs the platform consumes — historical bookings,
historical own-prices, occupancy snapshots, inventory, competitor listings,
local events, and holidays — from a pluggable :class:`DataSource`. The default
:class:`FileDataSource` reads canonically named files from ``data/raw`` (CSV for
tabular entities, JSON for the messy competitor listings). Missing files are
tolerated and yield empty frames with a logged warning, so the pipeline runs
end-to-end even with a partial drop.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.utils.exceptions import ETLError
from app.utils.logging import get_logger

logger = get_logger("etl")

# Canonical filenames within the raw data directory.
BOOKINGS_FILE = "bookings.csv"
PRICES_FILE = "price_observations.csv"
OCCUPANCY_FILE = "occupancy.csv"
INVENTORY_FILE = "inventory.csv"
COMPETITOR_FILE = "competitor_listings.json"
EVENTS_FILE = "events.csv"
HOLIDAYS_FILE = "holidays.csv"
OBSERVATIONS_FILE = "observations.csv"


@dataclass
class ExtractedData:
    """Container for the raw, unprocessed inputs pulled from a source.

    Tabular entities are pandas frames (possibly empty). Competitor listings are
    a list of loosely typed dicts mirroring scraper output, ready to be handed to
    the cleaning pipeline in the transform stage.
    """

    bookings: pd.DataFrame
    prices: pd.DataFrame
    occupancy: pd.DataFrame
    inventory: pd.DataFrame
    competitor_listings: list[dict[str, Any]]
    events: pd.DataFrame
    holidays: pd.DataFrame
    observations: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def row_counts(self) -> dict[str, int]:
        """Return a per-entity record count for logging and summaries."""
        return {
            "bookings": len(self.bookings),
            "prices": len(self.prices),
            "occupancy": len(self.occupancy),
            "inventory": len(self.inventory),
            "competitor_listings": len(self.competitor_listings),
            "events": len(self.events),
            "holidays": len(self.holidays),
            "observations": len(self.observations),
        }


class DataSource(ABC):
    """Abstract source of raw ETL inputs.

    Implementations return pandas frames for tabular entities and a list of
    dicts for competitor listings. Any method may return an empty result; the
    transform stage handles partial inputs gracefully.
    """

    @abstractmethod
    def read_bookings(self) -> pd.DataFrame:
        """Return historical bookings."""

    @abstractmethod
    def read_prices(self) -> pd.DataFrame:
        """Return historical own-rate observations."""

    @abstractmethod
    def read_occupancy(self) -> pd.DataFrame:
        """Return occupancy snapshots."""

    @abstractmethod
    def read_inventory(self) -> pd.DataFrame:
        """Return inventory by hotel/room/stay date."""

    @abstractmethod
    def read_competitor_listings(self) -> list[dict[str, Any]]:
        """Return raw, messy competitor listings."""

    @abstractmethod
    def read_events(self) -> pd.DataFrame:
        """Return the local-events calendar."""

    @abstractmethod
    def read_holidays(self) -> pd.DataFrame:
        """Return the holiday calendar."""

    @abstractmethod
    def read_observations(self) -> pd.DataFrame:
        """Return the consolidated training observation frame, if available."""


class FileDataSource(DataSource):
    """Reads raw inputs from canonically named files in a directory."""

    def __init__(self, raw_dir: Path | str) -> None:
        self._dir = Path(raw_dir)

    def _read_csv(self, name: str, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
        """Read a CSV if present, otherwise return an empty frame."""
        path = self._dir / name
        if not path.exists():
            logger.warning("raw file missing, skipping: {}", path)
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path, parse_dates=parse_dates or [])
        except Exception as exc:  # pragma: no cover - corrupt file path
            raise ETLError(
                "Failed to read raw CSV file.",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        logger.info("read {} rows from {}", len(frame), path.name)
        return frame

    def read_bookings(self) -> pd.DataFrame:
        """Read historical bookings from the raw directory."""
        return self._read_csv(BOOKINGS_FILE, parse_dates=["booking_date", "stay_date"])

    def read_prices(self) -> pd.DataFrame:
        """Read historical own-rate observations from the raw directory."""
        return self._read_csv(PRICES_FILE, parse_dates=["observed_on", "stay_date"])

    def read_occupancy(self) -> pd.DataFrame:
        """Read occupancy snapshots from the raw directory."""
        return self._read_csv(OCCUPANCY_FILE, parse_dates=["stay_date", "as_of"])

    def read_inventory(self) -> pd.DataFrame:
        """Read inventory rows from the raw directory."""
        return self._read_csv(INVENTORY_FILE, parse_dates=["stay_date"])

    def read_events(self) -> pd.DataFrame:
        """Read the local-events calendar from the raw directory."""
        return self._read_csv(EVENTS_FILE, parse_dates=["start_date", "end_date"])

    def read_holidays(self) -> pd.DataFrame:
        """Read the holiday calendar from the raw directory."""
        return self._read_csv(HOLIDAYS_FILE, parse_dates=["holiday_date"])

    def read_observations(self) -> pd.DataFrame:
        """Read the consolidated training observation frame from the raw directory."""
        return self._read_csv(OBSERVATIONS_FILE, parse_dates=["stay_date", "booking_date"])

    def read_competitor_listings(self) -> list[dict[str, Any]]:
        """Read raw competitor listings (JSON) from the raw directory."""
        path = self._dir / COMPETITOR_FILE
        if not path.exists():
            logger.warning("raw file missing, skipping: {}", path)
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt file path
            raise ETLError(
                "Failed to parse competitor listings JSON.",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        listings = payload if isinstance(payload, list) else payload.get("listings", [])
        logger.info("read {} competitor listings from {}", len(listings), path.name)
        return list(listings)


def extract(
    source: DataSource | None = None,
    *,
    settings: Settings | None = None,
) -> ExtractedData:
    """Run the extraction stage.

    Args:
        source: The data source to read from. Defaults to a
            :class:`FileDataSource` rooted at ``settings.raw_data_dir``.
        settings: Application settings (defaults to the cached singleton).

    Returns:
        An :class:`ExtractedData` bundle of all raw inputs.
    """
    settings = settings or get_settings()
    source = source or FileDataSource(settings.raw_data_dir)

    data = ExtractedData(
        bookings=source.read_bookings(),
        prices=source.read_prices(),
        occupancy=source.read_occupancy(),
        inventory=source.read_inventory(),
        competitor_listings=source.read_competitor_listings(),
        events=source.read_events(),
        holidays=source.read_holidays(),
        observations=source.read_observations(),
    )
    counts = data.row_counts()
    data.meta["row_counts"] = counts
    logger.info("extraction complete: {}", counts)
    return data
