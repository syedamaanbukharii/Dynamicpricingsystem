"""Competitor Cleaning Agent.

Transforms messy, scraper-emitted :class:`RawCompetitorListing` records into
clean, structured :class:`CompetitorRate` rows. Responsibilities:

* drop advertisements and malformed entries,
* extract a numeric price and detect its currency from heterogeneous formats
  (``"$189"``, ``"USD 205.00"``, ``"€175"``, ``"1,299"``),
* normalize room names to the canonical vocabulary (via the Room Matching Agent),
* validate fields, and
* de-duplicate identical offers.

Price/currency parsing is deterministic and regex-based; the LLM is consulted
only for strings the rules cannot parse and only when it is enabled, so the
agent works fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from app.agents.llm import LLMClient
from app.agents.room_matching import RoomMatcher
from app.schemas.common import Currency, RoomType
from app.schemas.competitor import CompetitorRate, RawCompetitorListing
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger

logger = get_logger("agent")

_CURRENCY_SYMBOLS = {"$": Currency.USD, "€": Currency.EUR, "£": Currency.GBP, "¥": Currency.JPY}
_CURRENCY_CODES = {c.value: c for c in Currency}
_PRICE_RE = re.compile(r"(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass
class CleaningResult:
    """Outcome of a cleaning run: clean rates plus a dropped-records audit."""

    rates: list[CompetitorRate] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    @property
    def kept(self) -> int:
        """Number of clean rates produced."""
        return len(self.rates)

    @property
    def removed(self) -> int:
        """Number of listings discarded during cleaning."""
        return len(self.dropped)


class CompetitorCleaningAgent:
    """Cleans and structures raw competitor listings."""

    def __init__(
        self,
        room_matcher: RoomMatcher | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._llm = llm
        self._matcher = room_matcher or RoomMatcher(llm=llm)

    @staticmethod
    def _detect_currency(text: str) -> Currency:
        """Infer the currency from a symbol or ISO code in the string."""
        for symbol, currency in _CURRENCY_SYMBOLS.items():
            if symbol in text:
                return currency
        upper = text.upper()
        for code, currency in _CURRENCY_CODES.items():
            if re.search(rf"\b{code}\b", upper):
                return currency
        return Currency.USD

    def _parse_price(self, raw_price: str) -> tuple[float, Currency] | None:
        """Return ``(price, currency)`` or ``None`` if unparseable."""
        if not raw_price or raw_price.strip().lower() in {"n/a", "na", "none", "-", ""}:
            return None
        currency = self._detect_currency(raw_price)
        match = _PRICE_RE.search(raw_price.replace(",", ""))
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                value = -1.0
            if value > 0:
                return value, currency
        return self._llm_parse_price(raw_price)

    def _llm_parse_price(self, raw_price: str) -> tuple[float, Currency] | None:
        """Last-resort price extraction via the LLM (when enabled)."""
        if self._llm is None or not self._llm.enabled:
            return None
        try:
            payload = self._llm.complete_json(
                system="You extract a numeric nightly price and ISO currency code.",
                user=(
                    f'Price text: "{raw_price}". '
                    'Return {"price": <number or null>, "currency": "USD|EUR|GBP|JPY|AUD|CAD|INR"}.'
                ),
            )
            price = payload.get("price")
            if price is None:
                return None
            currency = Currency(str(payload.get("currency", "USD")).upper())
            value = float(price)
            return (value, currency) if value > 0 else None
        except (LLMError, ValueError, TypeError) as exc:
            logger.warning("LLM price parse failed for {!r}: {}", raw_price, exc)
            return None

    def clean(
        self,
        listings: list[RawCompetitorListing],
        *,
        default_stay_date: date | None = None,
    ) -> CleaningResult:
        """Clean a batch of raw listings into structured competitor rates."""
        result = CleaningResult()
        seen: set[tuple[str, str, str, float]] = set()

        for listing in listings:
            if listing.is_advertisement or listing.raw_room_name.lower().startswith(
                ("sponsored", "ad:", "advertisement")
            ):
                result.dropped.append(
                    {"reason": "advertisement", "raw": listing.raw_room_name}
                )
                continue

            parsed = self._parse_price(listing.raw_price)
            if parsed is None:
                result.dropped.append(
                    {"reason": "unparseable_price", "raw": listing.raw_price}
                )
                continue
            price, currency = parsed

            stay = listing.stay_date or default_stay_date
            if stay is None:
                result.dropped.append(
                    {"reason": "missing_stay_date", "raw": listing.raw_room_name}
                )
                continue

            room_type = self._matcher.match(listing.raw_room_name)
            if room_type is RoomType.OTHER:
                logger.debug("unmatched room name -> OTHER: {!r}", listing.raw_room_name)

            dedup_key = (
                listing.competitor,
                room_type.value,
                stay.isoformat(),
                round(price, 2),
            )
            if dedup_key in seen:
                result.dropped.append(
                    {"reason": "duplicate", "raw": listing.raw_room_name}
                )
                continue
            seen.add(dedup_key)

            result.rates.append(
                CompetitorRate(
                    source=listing.source,
                    competitor=listing.competitor,
                    room_type=room_type,
                    raw_room_name=listing.raw_room_name,
                    price=round(price, 2),
                    currency=currency,
                    stay_date=stay,
                    scraped_at=listing.scraped_at,
                )
            )

        logger.info(
            "competitor cleaning: kept={} dropped={}", result.kept, result.removed
        )
        return result
