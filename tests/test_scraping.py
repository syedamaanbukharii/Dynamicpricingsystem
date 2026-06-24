"""Tests for the scraping layer: selector-driven parsing and incremental skip."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from app.config import get_settings
from app.scraping.base import BaseScraper, ScrapeStateStore
from app.scraping.targets import ScrapeSelectors, ScrapeTarget

_HTML = """
<html><body>
  <div class="room">
    <span class="name">Deluxe King Room</span>
    <span class="price">$199</span>
  </div>
  <div class="room">
    <span class="name">Standard Queen</span>
    <span class="price">USD 149.00</span>
  </div>
  <div class="ad"><span class="name">Sponsored: Deal</span><span class="price">$1</span></div>
</body></html>
"""


class FakeScraper(BaseScraper):
    """A scraper whose fetch returns canned HTML, for offline testing."""

    def __init__(self, target: ScrapeTarget, settings) -> None:
        super().__init__(target, settings)
        self.fetch_calls = 0

    async def _fetch_html(self, url: str) -> str:
        self.fetch_calls += 1
        return _HTML

    async def _robots_allows(self, url: str) -> bool:  # bypass network robots check
        return True


def _target() -> ScrapeTarget:
    return ScrapeTarget(
        name="fake_target",
        competitor="Fake Hotel",
        base_url="https://example.com",
        search_url_template="https://example.com/search?d={stay_date}",
        selectors=ScrapeSelectors(listing="div.room", room_name="span.name", price="span.price"),
        requires_javascript=False,
    )


@pytest.mark.asyncio
async def test_parse_extracts_listings() -> None:
    """Selector-driven parsing extracts one listing per matching container."""
    settings = get_settings()
    async with FakeScraper(_target(), settings) as scraper:
        listings = await scraper.scrape([date.today() + timedelta(days=3)], incremental=False)
    # Two .room containers parse (the .ad container is not matched by div.room).
    assert len(listings) == 2
    names = {listing.raw_room_name for listing in listings}
    assert "Deluxe King Room" in names


@pytest.mark.asyncio
async def test_incremental_skips_recent(tmp_path: Path) -> None:
    """A second incremental run within the refresh window skips the stay date."""
    settings = get_settings()
    state = ScrapeStateStore(tmp_path / "state.json")
    stay = [date.today() + timedelta(days=4)]

    async with FakeScraper(_target(), settings) as scraper:
        first = await scraper.scrape(
            stay, incremental=True, state_store=state, min_refresh_seconds=10_000
        )
    state.flush()
    assert len(first) == 2

    async with FakeScraper(_target(), settings) as scraper:
        second = await scraper.scrape(
            stay, incremental=True, state_store=state, min_refresh_seconds=10_000
        )
    assert second == []
