"""Scraper foundation: rate limiting, retries, robots, and parsing.

:class:`BaseScraper` owns everything that is identical across crawl backends --
URL templating, polite rate limiting, timeout-bounded retried fetches, optional
``robots.txt`` enforcement, incremental-scrape bookkeeping, and selector-driven
HTML parsing into :class:`RawCompetitorListing` records. Concrete backends only
implement :meth:`_fetch_html`. No selectors are hardcoded here; they come from
the :class:`ScrapeTarget` configuration.
"""

from __future__ import annotations

import abc
import asyncio
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

from app.config import Settings, get_settings
from app.schemas.competitor import RawCompetitorListing
from app.scraping.targets import ScrapeTarget
from app.utils.exceptions import ScrapingError
from app.utils.logging import get_logger
from app.utils.retry import async_retry

logger = get_logger("scraper")


class RateLimiter:
    """Async rate limiter enforcing a minimum delay between requests."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Block until at least ``min_interval`` has elapsed since the last call."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


class ScrapeStateStore:
    """Persists last-scrape timestamps to support incremental crawling."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state: dict[str, str] = {}
        if path.exists():
            try:
                self._state = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def last_scraped(self, key: str) -> datetime | None:
        """Return the last scrape time for a key, if recorded."""
        raw = self._state.get(key)
        return datetime.fromisoformat(raw) if raw else None

    def mark(self, key: str) -> None:
        """Record the current time as the last scrape time for a key."""
        self._state[key] = datetime.now(UTC).isoformat()

    def flush(self) -> None:
        """Persist the in-memory state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))


class BaseScraper(abc.ABC):
    """Abstract base for competitor scrapers."""

    def __init__(
        self,
        target: ScrapeTarget,
        settings: Settings | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.target = target
        self.settings = settings or get_settings()
        self.rate_limiter = rate_limiter or RateLimiter(
            self.settings.scrape_request_delay_seconds
        )
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    @abc.abstractmethod
    async def _fetch_html(self, url: str) -> str:
        """Fetch the rendered HTML for a URL (backend-specific)."""
        raise NotImplementedError

    async def _robots_allows(self, url: str) -> bool:
        """Return whether ``robots.txt`` permits crawling ``url``."""
        if not self.settings.scrape_respect_robots:
            return True
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._robots.get(host)
        if parser is None:
            parser = robotparser.RobotFileParser()
            parser.set_url(f"{host}/robots.txt")
            try:
                await asyncio.to_thread(parser.read)
            except Exception as exc:
                logger.warning("robots.txt fetch failed for {}: {}", host, exc)
                self._robots[host] = parser
                return True
            self._robots[host] = parser
        return parser.can_fetch(self.settings.scrape_user_agent, url)

    async def _fetch_with_policies(self, url: str) -> str:
        """Apply rate limiting, timeout, and retry around the raw fetch."""
        await self.rate_limiter.wait()

        @async_retry(
            attempts=self.settings.scrape_max_retries,
            base_delay=self.settings.scrape_request_delay_seconds,
            exceptions=(ScrapingError, asyncio.TimeoutError),
        )
        async def _do() -> str:
            try:
                return await asyncio.wait_for(
                    self._fetch_html(url), timeout=self.settings.scrape_timeout_seconds
                )
            except TimeoutError:
                raise
            except ScrapingError:
                raise
            except Exception as exc:
                raise ScrapingError(
                    "Fetch failed.", details={"url": url, "error": str(exc)}
                ) from exc

        return await _do()

    def _parse(self, html: str, stay_date: date) -> list[RawCompetitorListing]:
        """Parse listings from HTML using the target's configured selectors."""
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - parsing dep missing
            raise ScrapingError(
                "beautifulsoup4 is required to parse scraped HTML.",
                details={"error": str(exc)},
            ) from exc

        soup = BeautifulSoup(html, "html.parser")
        selectors = self.target.selectors
        listings: list[RawCompetitorListing] = []
        for node in soup.select(selectors.listing):
            name_el = node.select_one(selectors.room_name)
            price_el = node.select_one(selectors.price)
            if name_el is None or price_el is None:
                continue
            listings.append(
                RawCompetitorListing(
                    source=self.target.name,
                    competitor=self.target.competitor,
                    raw_room_name=name_el.get_text(strip=True),
                    raw_price=price_el.get_text(strip=True),
                    stay_date=stay_date,
                    url=self.target.url_for(stay_date.isoformat()),
                )
            )
        logger.info(
            "parsed {} listings from {} for {}",
            len(listings),
            self.target.name,
            stay_date.isoformat(),
        )
        return listings

    async def scrape(
        self,
        stay_dates: list[date],
        *,
        incremental: bool = True,
        state_store: ScrapeStateStore | None = None,
        min_refresh_seconds: float | None = None,
    ) -> list[RawCompetitorListing]:
        """Scrape the target for each stay date, returning raw listings."""
        refresh = (
            min_refresh_seconds
            if min_refresh_seconds is not None
            else float(self.settings.cache_ttl_seconds)
        )
        results: list[RawCompetitorListing] = []
        for stay_date in stay_dates:
            key = f"{self.target.name}:{stay_date.isoformat()}"
            if incremental and state_store is not None:
                last = state_store.last_scraped(key)
                if last is not None:
                    age = (datetime.now(UTC) - last).total_seconds()
                    if age < refresh:
                        logger.debug("skip {} (scraped {:.0f}s ago)", key, age)
                        continue

            url = self.target.url_for(stay_date.isoformat())
            if not await self._robots_allows(url):
                logger.warning("robots.txt disallows {}; skipping", url)
                continue

            html = await self._fetch_with_policies(url)
            results.extend(self._parse(html, stay_date))
            if state_store is not None:
                state_store.mark(key)

        if state_store is not None:
            state_store.flush()
        return results

    async def __aenter__(self) -> BaseScraper:
        """Enter the async context (override to acquire resources)."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context (override to release resources)."""
        return None
