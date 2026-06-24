"""Data Quality Agent.

Runs deterministic quality checks over cleaned competitor rates and surfaces a
structured :class:`QualityReport`. Detects missing/unmatched fields, invalid
prices, duplicate records, currency mismatches, timestamp anomalies, and price
outliers (per room type, via the IQR rule). The report is advisory: callers
decide whether to proceed, alert, or quarantine, keeping the agent side-effect
free and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

import numpy as np

from app.schemas.common import RoomType, utcnow
from app.schemas.competitor import CompetitorRate
from app.utils.logging import get_logger

logger = get_logger("agent")


class Severity(str, Enum):
    """Issue severity used to gate downstream processing."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityIssue:
    """A single detected data-quality problem with example offenders."""

    kind: str
    severity: Severity
    message: str
    count: int
    samples: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Aggregated quality findings for a batch of records."""

    total_records: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Whether any error-severity issue was found."""
        return any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def is_ok(self) -> bool:
        """True when no warning- or error-severity issues are present."""
        return not any(i.severity in {Severity.WARNING, Severity.ERROR} for i in self.issues)

    def to_dict(self) -> dict[str, object]:
        """Serialize the report for API responses or logging."""
        return {
            "total_records": self.total_records,
            "has_errors": self.has_errors,
            "is_ok": self.is_ok,
            "issues": [
                {
                    "kind": i.kind,
                    "severity": i.severity.value,
                    "message": i.message,
                    "count": i.count,
                    "samples": i.samples,
                }
                for i in self.issues
            ],
        }


class DataQualityAgent:
    """Detects quality problems in cleaned competitor rates."""

    def __init__(self, *, max_plausible_price: float = 100_000.0) -> None:
        self._max_price = max_plausible_price

    def assess(self, rates: list[CompetitorRate]) -> QualityReport:
        """Evaluate a batch of rates and return a structured report."""
        report = QualityReport(total_records=len(rates))
        if not rates:
            report.issues.append(
                QualityIssue("empty", Severity.WARNING, "No records to assess.", 0)
            )
            return report

        self._check_unmatched(rates, report)
        self._check_prices(rates, report)
        self._check_duplicates(rates, report)
        self._check_currency(rates, report)
        self._check_timestamps(rates, report)
        self._check_outliers(rates, report)

        logger.info(
            "data quality: {} records, {} issues ({} errors)",
            report.total_records,
            len(report.issues),
            sum(1 for i in report.issues if i.severity is Severity.ERROR),
        )
        return report

    @staticmethod
    def _check_unmatched(rates: list[CompetitorRate], report: QualityReport) -> None:
        unmatched = [r.raw_room_name for r in rates if r.room_type is RoomType.OTHER]
        if unmatched:
            report.issues.append(
                QualityIssue(
                    "unmatched_room_type",
                    Severity.WARNING,
                    "Some room names could not be mapped to a canonical type.",
                    len(unmatched),
                    unmatched[:5],
                )
            )

    def _check_prices(self, rates: list[CompetitorRate], report: QualityReport) -> None:
        invalid = [r.raw_room_name for r in rates if r.price <= 0 or r.price > self._max_price]
        if invalid:
            report.issues.append(
                QualityIssue(
                    "invalid_price",
                    Severity.ERROR,
                    "Prices outside the plausible range were detected.",
                    len(invalid),
                    invalid[:5],
                )
            )

    @staticmethod
    def _check_duplicates(rates: list[CompetitorRate], report: QualityReport) -> None:
        seen: set[tuple[str, str, str, float]] = set()
        dupes: list[str] = []
        for r in rates:
            key = (r.competitor, r.room_type.value, r.stay_date.isoformat(), r.price)
            if key in seen:
                dupes.append(f"{r.competitor}:{r.room_type.value}:{r.stay_date}")
            seen.add(key)
        if dupes:
            report.issues.append(
                QualityIssue(
                    "duplicate_record",
                    Severity.WARNING,
                    "Duplicate competitor offers remain in the batch.",
                    len(dupes),
                    dupes[:5],
                )
            )

    @staticmethod
    def _check_currency(rates: list[CompetitorRate], report: QualityReport) -> None:
        by_group: dict[tuple[str, str], set[str]] = {}
        for r in rates:
            by_group.setdefault((r.competitor, r.stay_date.isoformat()), set()).add(
                r.currency.value
            )
        mixed = [f"{c}@{d}" for (c, d), cur in by_group.items() if len(cur) > 1]
        if mixed:
            report.issues.append(
                QualityIssue(
                    "currency_mismatch",
                    Severity.WARNING,
                    "Multiple currencies for the same competitor/date.",
                    len(mixed),
                    mixed[:5],
                )
            )

    @staticmethod
    def _check_timestamps(rates: list[CompetitorRate], report: QualityReport) -> None:
        now = utcnow()
        today = now.date()
        future_scrapes = [
            r.raw_room_name for r in rates if r.scraped_at > now + timedelta(minutes=5)
        ]
        stale_stays = [r.raw_room_name for r in rates if r.stay_date < today - timedelta(days=1)]
        if future_scrapes:
            report.issues.append(
                QualityIssue(
                    "future_timestamp",
                    Severity.ERROR,
                    "Records have scrape timestamps in the future.",
                    len(future_scrapes),
                    future_scrapes[:5],
                )
            )
        if stale_stays:
            report.issues.append(
                QualityIssue(
                    "stale_stay_date",
                    Severity.INFO,
                    "Records reference stay dates in the past.",
                    len(stale_stays),
                    stale_stays[:5],
                )
            )

    @staticmethod
    def _check_outliers(rates: list[CompetitorRate], report: QualityReport) -> None:
        by_type: dict[RoomType, list[CompetitorRate]] = {}
        for r in rates:
            by_type.setdefault(r.room_type, []).append(r)
        outliers: list[str] = []
        for room_type, group in by_type.items():
            prices = np.array([g.price for g in group], dtype=float)
            if prices.size < 4:
                continue
            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers.extend(
                f"{room_type.value}:{g.competitor}:{g.price}"
                for g in group
                if g.price < low or g.price > high
            )
        if outliers:
            report.issues.append(
                QualityIssue(
                    "price_outlier",
                    Severity.WARNING,
                    "Prices deviate sharply from peers (IQR rule).",
                    len(outliers),
                    outliers[:5],
                )
            )
