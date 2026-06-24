"""Web scraping package: configurable, polite, selector-driven scrapers."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.scraping.base import BaseScraper, RateLimiter, ScrapeStateStore
from app.scraping.crawl4ai_scraper import Crawl4AIScraper
from app.scraping.playwright_scraper import PlaywrightScraper
from app.scraping.targets import (
    ScrapeConfig,
    ScrapeSelectors,
    ScrapeTarget,
    load_targets,
)


def build_scraper(target: ScrapeTarget, settings: Settings | None = None) -> BaseScraper:
    """Return an appropriate scraper backend for the target.

    JavaScript-dependent targets use the Playwright renderer; static targets use
    the lighter Crawl4AI backend.
    """
    settings = settings or get_settings()
    if target.requires_javascript:
        return PlaywrightScraper(target, settings)
    return Crawl4AIScraper(target, settings)


__all__ = [
    "BaseScraper",
    "Crawl4AIScraper",
    "PlaywrightScraper",
    "RateLimiter",
    "ScrapeConfig",
    "ScrapeSelectors",
    "ScrapeStateStore",
    "ScrapeTarget",
    "build_scraper",
    "load_targets",
]
