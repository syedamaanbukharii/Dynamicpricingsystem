"""Synthetic data generation for local development, demos, and tests.

This module simulates a plausible booking history with a genuine downward demand
curve (so a model can learn price sensitivity), plus messy raw competitor
listings for exercising the cleaning agents. It is a developer utility -- not
production logic -- and is never imported by the running service.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pandas as pd
from app.feature_engineering.calendars import EventCalendar, HolidayCalendar
from app.feature_engineering.defaults import default_events, default_holidays
from app.schemas.common import RoomType

_BASE_PRICE = {
    RoomType.STANDARD_QUEEN: 120.0,
    RoomType.STANDARD_KING: 135.0,
    RoomType.DELUXE_KING: 175.0,
    RoomType.JUNIOR_SUITE: 240.0,
    RoomType.EXECUTIVE_SUITE: 330.0,
}
_INVENTORY = {
    RoomType.STANDARD_QUEEN: 60,
    RoomType.STANDARD_KING: 50,
    RoomType.DELUXE_KING: 40,
    RoomType.JUNIOR_SUITE: 20,
    RoomType.EXECUTIVE_SUITE: 10,
}


def generate_training_frame(
    *,
    start: date = date(2024, 1, 1),
    days: int = 540,
    hotels: tuple[str, ...] = ("HOTEL_A", "HOTEL_B"),
    room_types: tuple[RoomType, ...] = tuple(_BASE_PRICE),
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a realistic booking-history frame with a learnable demand curve."""
    rng = random.Random(seed)
    holidays = HolidayCalendar(default_holidays())
    events = EventCalendar(default_events())
    rows: list[dict[str, object]] = []

    for offset in range(days):
        stay_date = start + timedelta(days=offset)
        seasonal = 1.0 + 0.25 * math.sin(2 * math.pi * (stay_date.timetuple().tm_yday / 365.0))
        weekend = 1.12 if stay_date.weekday() >= 5 else 1.0
        holiday = 1.18 if holidays.is_holiday(stay_date) else 1.0
        event = 1.0 + events.event_score(stay_date)

        for hotel in hotels:
            hotel_factor = 1.0 if hotel == "HOTEL_A" else 0.92
            for room in room_types:
                base = _BASE_PRICE[room] * hotel_factor
                inventory = _INVENTORY[room]
                comp_median = base * rng.uniform(0.92, 1.12)
                comp_min = comp_median * rng.uniform(0.82, 0.95)
                comp_max = comp_median * rng.uniform(1.05, 1.25)
                lead = rng.randint(1, 60)
                # Explore a wide price band (well above and below the competitor
                # median) so the model observes genuine price elasticity and the
                # pricing engine can locate an interior revenue optimum rather
                # than extrapolating flat beyond the training support.
                price = comp_median * rng.uniform(0.55, 1.85)

                ratio = price / comp_median
                demand_base = 0.78 * seasonal * weekend * holiday * event * hotel_factor
                fraction = demand_base * math.exp(-1.9 * (ratio - 1.0))
                fraction *= rng.uniform(0.9, 1.1)  # idiosyncratic noise
                fraction = max(0.0, min(1.0, fraction))
                rooms_sold = round(fraction * inventory)

                rows.append(
                    {
                        "hotel_id": hotel,
                        "room_type": room.value,
                        "stay_date": stay_date,
                        "booking_date": stay_date - timedelta(days=lead),
                        "price": round(price, 2),
                        "previous_price": round(price * rng.uniform(0.95, 1.05), 2),
                        "inventory_total": inventory,
                        "rooms_sold": rooms_sold,
                        "competitor_median": round(comp_median, 2),
                        "competitor_min": round(comp_min, 2),
                        "competitor_max": round(comp_max, 2),
                        "booking_velocity": round(rooms_sold / max(lead, 1) * 7, 3),
                        "cancellation_rate": round(rng.uniform(0.02, 0.15), 3),
                    }
                )
    return pd.DataFrame(rows)


def generate_raw_competitor_listings(seed: int = 7) -> list[dict[str, object]]:
    """Return messy, scraper-like competitor listings for cleaning-agent demos."""
    rng = random.Random(seed)
    today = date.today()
    names = [
        "Deluxe King Room",
        "King Deluxe",
        "Premium King Roo",
        "DELUXE KING SUITE",
        "Standard Queen",
        "Queen Standard Room",
        "Std. Queen",
        "Sponsored: Best Deal!",
    ]
    listings: list[dict[str, object]] = [
        {
            "source": "ota_demo",
            "competitor": rng.choice(["Grand Plaza", "City Inn", "Harbor Hotel"]),
            "raw_room_name": name,
            "raw_price": rng.choice(["$189", "USD 205.00", "€175", "210", "n/a"]),
            "stay_date": (today + timedelta(days=rng.randint(1, 30))).isoformat(),
            "url": "https://example.com/listing",
            "is_advertisement": name.lower().startswith("sponsored"),
        }
        for name in names
    ]
    return listings
