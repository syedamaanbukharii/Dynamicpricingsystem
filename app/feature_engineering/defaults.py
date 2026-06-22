"""Default holiday and event calendars used across training and serving.

Keeping a single source of truth ensures the holiday/event features a model is
trained on match those produced at inference time. In production these would be
sourced from the ETL holiday/event tables; the defaults below provide a sensible
US-centric baseline so the system is runnable out of the box.
"""

from __future__ import annotations

from datetime import date

from app.feature_engineering.calendars import EventCalendar, HolidayCalendar
from app.feature_engineering.features import FeatureBuilder
from app.schemas.events import Holiday, LocalEvent

_FIXED_HOLIDAYS = [
    ("New Year's Day", (1, 1)),
    ("Independence Day", (7, 4)),
    ("Veterans Day", (11, 11)),
    ("Christmas Eve", (12, 24)),
    ("Christmas Day", (12, 25)),
    ("New Year's Eve", (12, 31)),
]

# Observed/derived holidays per year (month, day) computed for the baseline set.
_DATED_HOLIDAYS: dict[int, list[tuple[str, tuple[int, int]]]] = {
    2024: [("Memorial Day", (5, 27)), ("Labor Day", (9, 2)), ("Thanksgiving", (11, 28))],
    2025: [("Memorial Day", (5, 26)), ("Labor Day", (9, 1)), ("Thanksgiving", (11, 27))],
    2026: [("Memorial Day", (5, 25)), ("Labor Day", (9, 7)), ("Thanksgiving", (11, 26))],
    2027: [("Memorial Day", (5, 31)), ("Labor Day", (9, 6)), ("Thanksgiving", (11, 25))],
}


def default_holidays(years: tuple[int, ...] = (2024, 2025, 2026, 2027)) -> list[Holiday]:
    """Return the baseline holiday list across the requested years."""
    holidays: list[Holiday] = []
    for year in years:
        for name, (month, day) in _FIXED_HOLIDAYS:
            holidays.append(Holiday(name=name, holiday_date=date(year, month, day)))
        for name, (month, day) in _DATED_HOLIDAYS.get(year, []):
            holidays.append(Holiday(name=name, holiday_date=date(year, month, day)))
    return holidays


def default_events(years: tuple[int, ...] = (2024, 2025, 2026, 2027)) -> list[LocalEvent]:
    """Return a small baseline of recurring demand-affecting local events."""
    events: list[LocalEvent] = []
    for year in years:
        events.append(
            LocalEvent(
                name="City Tech Conference",
                start_date=date(year, 6, 15),
                end_date=date(year, 6, 18),
                expected_attendance=20000,
                demand_multiplier=1.6,
            )
        )
        events.append(
            LocalEvent(
                name="Autumn Marathon",
                start_date=date(year, 10, 10),
                end_date=date(year, 10, 11),
                expected_attendance=8000,
                demand_multiplier=1.3,
            )
        )
    return events


def default_feature_builder() -> FeatureBuilder:
    """Construct a :class:`FeatureBuilder` wired with the default calendars."""
    return FeatureBuilder(
        holidays=HolidayCalendar(default_holidays()),
        events=EventCalendar(default_events()),
    )
