"""Crawl4AI-backed scraper.

Uses Crawl4AI's managed asynchronous crawler to retrieve page HTML (handling
fetching, caching, and content extraction). Suitable for sources that do not
require full browser rendering, or where Crawl4AI's extraction pipeline is
preferred. The ``crawl4ai`` import is deferred so importing this module never
requires the dependency.
"""

from __future__ import annotations

from typing import Any

from app.scraping.base import BaseScraper
from app.utils.exceptions import ScrapingError
from app.utils.logging import get_logger

logger = get_logger("scraper")


class Crawl4AIScraper(BaseScraper):
    """Fetch HTML using the Crawl4AI asynchronous crawler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._crawler: Any | None = None

    async def _ensure_crawler(self) -> Any:
        """Construct and start the Crawl4AI crawler once."""
        if self._crawler is not None:
            return self._crawler
        try:
            from crawl4ai import AsyncWebCrawler
        except ImportError as exc:  # pragma: no cover - crawler dep missing
            raise ScrapingError(
                "crawl4ai is required for Crawl4AIScraper.",
                details={"error": str(exc)},
            ) from exc
        self._crawler = AsyncWebCrawler(verbose=False)
        await self._crawler.start()
        logger.info("started crawl4ai crawler for {}", self.target.name)
        return self._crawler

    async def _fetch_html(self, url: str) -> str:
        """Crawl ``url`` and return the retrieved HTML."""
        crawler = await self._ensure_crawler()
        result = await crawler.arun(url=url, user_agent=self.settings.scrape_user_agent)
        html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
        if not html:
            raise ScrapingError("Crawl4AI returned no HTML.", details={"url": url})
        return html

    async def __aexit__(self, *exc: object) -> None:
        """Stop the Crawl4AI crawler."""
        if self._crawler is not None:
            close = getattr(self._crawler, "close", None) or getattr(self._crawler, "stop", None)
            if close is not None:
                await close()
            self._crawler = None
