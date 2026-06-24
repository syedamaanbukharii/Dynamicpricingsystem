r"""CLI: produce a single price recommendation from the command line.

Usage:
    python -m scripts.run_prediction --hotel HOTEL_A --room DELUXE_KING \\
        --stay 2026-07-15 --inventory 50 --sold 20 --previous-price 180 \\
        --competitors 175,189,205,210

Runs against a trained model if one exists, otherwise the heuristic fallback.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

from app.schemas.common import RoomType
from app.schemas.pricing import PriceRecommendationRequest
from app.services.pricing_service import get_pricing_service
from app.utils.logging import configure_logging


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    """Entry point for the prediction CLI."""
    configure_logging()
    default_stay = (date.today() + timedelta(days=21)).isoformat()

    parser = argparse.ArgumentParser(description="Recommend a nightly price.")
    parser.add_argument("--hotel", default="HOTEL_A", help="Hotel identifier.")
    parser.add_argument(
        "--room",
        default="DELUXE_KING",
        choices=[rt.value for rt in RoomType],
        help="Room type.",
    )
    parser.add_argument("--stay", default=default_stay, help="Stay date (YYYY-MM-DD).")
    parser.add_argument("--inventory", type=int, default=50, help="Total inventory.")
    parser.add_argument("--sold", type=int, default=20, help="Rooms already sold.")
    parser.add_argument(
        "--previous-price", type=float, default=None, help="Yesterday's published price."
    )
    parser.add_argument(
        "--competitors",
        default="",
        help="Comma-separated competitor nightly rates, e.g. 175,189,205.",
    )
    parser.add_argument("--velocity", type=float, default=None, help="Recent bookings/day pace.")
    args = parser.parse_args()

    competitors = (
        [float(x) for x in args.competitors.split(",") if x.strip()] if args.competitors else None
    )

    request = PriceRecommendationRequest(
        hotel_id=args.hotel,
        room_type=RoomType(args.room),
        stay_date=_parse_date(args.stay),
        inventory_total=args.inventory,
        rooms_sold=args.sold,
        previous_price=args.previous_price,
        competitor_rates=competitors,
        booking_velocity=args.velocity,
        include_explanation=True,
    )

    service = get_pricing_service()
    response = service.recommend(request)
    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
