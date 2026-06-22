"""Calendar schemas for holidays and local events."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Holiday(BaseModel):
    """A public or observed holiday for a market."""

    name: str
    holiday_date: date
    country: str = "US"
    region: str | None = None
    is_public: bool = True


class LocalEvent(BaseModel):
    """A demand-affecting local event (conference, concert, sports)."""

    name: str
    start_date: date
    end_date: date
    venue: str | None = None
    expected_attendance: int = Field(default=0, ge=0)
    demand_multiplier: float = Field(default=1.0, ge=0.0)

    def covers(self, day: date) -> bool:
        """Whether the event spans the given date."""
        return self.start_date <= day <= self.end_date
