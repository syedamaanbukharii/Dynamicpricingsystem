"""Playwright-backed scraper for JavaScript-rendered pages.

Renders pages in a headless Chromium instance so client-side-rendered prices are
visible before parsing. The browser is launched lazily and reused across the
scraper's lifetime; use the scraper as an async context manager to guarantee the
browser is closed. The heavy ``playwright`` import is deferred so importing this
module never requires the browser runtime.
"""

from __future__ import annotations

from typing import Any

from app.scraping.base import BaseScraper
from app.utils.exceptions import ScrapingError
from app.utils.logging import get_logger

logger = get_logger("scraper")


class PlaywrightScraper(BaseScraper):
    """Fetch fully-rendered HTML using a headless Chromium browser."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._playwright: Any | None = None
        self._browser: Any | None = None

    async def _ensure_browser(self) -> Any:
        """Launch Chromium once and reuse it for subsequent fetches."""
        if self._browser is not None:
            return self._browser
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - browser dep missing
            raise ScrapingError(
                "playwright is required for PlaywrightScraper. Install it and run "
                "'playwright install chromium'.",
                details={"error": str(exc)},
            ) from exc
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        logger.info("launched headless chromium for {}", self.target.name)
        return self._browser

    async def _fetch_html(self, url: str) -> str:
        """Render ``url`` and return its HTML content."""
        browser = await self._ensure_browser()
        context = await browser.new_context(user_agent=self.settings.scrape_user_agent)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle")
            return await page.content()
        finally:
            await context.close()

    async def __aexit__(self, *exc: object) -> None:
        """Close the browser and Playwright runtime."""
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
