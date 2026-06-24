"""Calendar helpers: season resolution, holiday proximity, and event scoring.

These are deliberately pure and side-effect free so the same logic runs both at
training time (vectorized over a DataFrame) and at inference time (single row).
"""

from __future__ import annotations

from datetime import date, timedelta

from app.schemas.common import Season
from app.schemas.events import Holiday, LocalEvent

_SEASON_BY_MONTH: dict[int, Season] = {
    12: Season.WINTER,
    1: Season.WINTER,
    2: Season.WINTER,
    3: Season.SPRING,
    4: Season.SPRING,
    5: Season.SPRING,
    6: Season.SUMMER,
    7: Season.SUMMER,
    8: Season.SUMMER,
    9: Season.AUTUMN,
    10: Season.AUTUMN,
    11: Season.AUTUMN,
}


def season_of(month: int) -> Season:
    """Map a calendar month (1..12) to a :class:`Season`."""
    if month < 1 or month > 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return _SEASON_BY_MONTH[month]


class HolidayCalendar:
    """Fast lookups for holiday membership and proximity."""

    def __init__(self, holidays: list[Holiday] | None = None) -> None:
        self._dates: set[date] = {h.holiday_date for h in (holidays or [])}

    @classmethod
    def from_dates(cls, dates: list[date]) -> HolidayCalendar:
        """Construct directly from a list of dates."""
        inst = cls()
        inst._dates = set(dates)
        return inst

    def is_holiday(self, day: date) -> bool:
        """Whether ``day`` is a holiday."""
        return day in self._dates

    def days_to_next_holiday(self, day: date, horizon: int = 120) -> int:
        """Days from ``day`` to the next holiday within ``horizon`` (else horizon)."""
        if not self._dates:
            return horizon
        for offset in range(horizon + 1):
            if (day + timedelta(days=offset)) in self._dates:
                return offset
        return horizon


class EventCalendar:
    """Local-event lookups and a bounded demand score for a date."""

    def __init__(self, events: list[LocalEvent] | None = None) -> None:
        self._events: list[LocalEvent] = list(events or [])

    def event_on(self, day: date) -> LocalEvent | None:
        """Return the strongest event covering ``day``, if any."""
        covering = [e for e in self._events if e.covers(day)]
        if not covering:
            return None
        return max(covering, key=lambda e: e.demand_multiplier)

    def is_event(self, day: date) -> bool:
        """Whether any event covers ``day``."""
        return self.event_on(day) is not None

    def event_score(self, day: date) -> float:
        """Bounded (0..1) demand contribution from events on ``day``."""
        event = self.event_on(day)
        if event is None:
            return 0.0
        # Convert a multiplier (>=1 amplifies) into a 0..1 score.
        return max(0.0, min(1.0, event.demand_multiplier - 1.0))
