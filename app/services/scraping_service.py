"""Scraping service.

Coordinates competitor scraping across configured targets and folds the raw
output through the cleaning + quality agents to yield structured competitor
rates. Network backends (Playwright/Crawl4AI) are import-guarded inside the
scraper classes, so this module imports cleanly even when they are absent; an
attempt to actually scrape without them raises a clear ``ScrapingError``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.agents.competitor_cleaning import CompetitorCleaningAgent
from app.agents.data_quality import DataQualityAgent, QualityReport
from app.agents.llm import get_llm_client
from app.agents.room_matching import RoomMatcher
from app.config import Settings, get_settings
from app.monitoring import record_scrape
from app.schemas.competitor import CompetitorRate, RawCompetitorListing
from app.scraping import ScrapeStateStore, build_scraper, load_targets
from app.utils.exceptions import ScrapingError
from app.utils.logging import get_logger

logger = get_logger("scraper")


@dataclass
class ScrapeServiceResult:
    """Outcome of a scrape + clean run."""

    rates: list[CompetitorRate] = field(default_factory=list)
    raw_count: int = 0
    quality: QualityReport | None = None
    targets_scraped: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "rates": len(self.rates),
            "raw_count": self.raw_count,
            "quality": self.quality.to_dict() if self.quality else None,
            "targets_scraped": self.targets_scraped,
            "errors": self.errors,
        }


async def scrape_and_clean(
    stay_dates: list[date],
    *,
    settings: Settings | None = None,
    target_names: list[str] | None = None,
    incremental: bool = True,
) -> ScrapeServiceResult:
    """Scrape configured targets for given stay dates and clean the results.

    Args:
        stay_dates: Stay dates to scrape rates for.
        settings: Application settings (defaults to the cached singleton).
        target_names: Restrict to these target names; default is all enabled.
        incremental: Skip dates scraped more recently than the refresh window.

    Returns:
        A :class:`ScrapeServiceResult` with cleaned rates and a quality report.
    """
    settings = settings or get_settings()
    config = load_targets(settings.scrape_targets_file)
    targets = config.enabled_targets()
    if target_names is not None:
        wanted = set(target_names)
        targets = [t for t in targets if t.name in wanted]
    if not targets:
        logger.warning("no scrape targets selected/enabled")
        return ScrapeServiceResult()

    state_store = ScrapeStateStore(
        Path(settings.processed_data_dir) / "scrape_state.json"
    )

    result = ScrapeServiceResult()
    raw_listings: list[RawCompetitorListing] = []

    for target in targets:
        try:
            async with build_scraper(target, settings) as scraper:
                listings = await scraper.scrape(
                    stay_dates, incremental=incremental, state_store=state_store
                )
            raw_listings.extend(listings)
            result.targets_scraped.append(target.name)
            record_scrape(target.name, "success")
            logger.info("scraped {} listings from {}", len(listings), target.name)
        except ScrapingError as exc:
            record_scrape(target.name, "failure")
            result.errors.append({"target": target.name, "error": str(exc)})
            logger.warning("scrape failed for {}: {}", target.name, exc)
        except Exception as exc:
            record_scrape(target.name, "failure")
            result.errors.append({"target": target.name, "error": str(exc)})
            logger.warning("scrape error for {}: {}", target.name, exc)

    state_store.flush()
    result.raw_count = len(raw_listings)

    llm = get_llm_client(settings)
    cleaner = CompetitorCleaningAgent(room_matcher=RoomMatcher(llm=llm), llm=llm)
    default_stay = stay_dates[0] if stay_dates else None
    cleaning = cleaner.clean(raw_listings, default_stay_date=default_stay)
    result.rates = cleaning.rates
    result.quality = DataQualityAgent().assess(cleaning.rates)
    logger.info(
        "scrape+clean produced {} rates from {} raw listings",
        len(result.rates),
        result.raw_count,
    )
    return result


def run_scrape(
    stay_dates: list[date],
    *,
    settings: Settings | None = None,
    target_names: list[str] | None = None,
    incremental: bool = True,
) -> ScrapeServiceResult:
    """Synchronous wrapper around :func:`scrape_and_clean` for CLIs/background.

    Runs the async scrape on a fresh event loop and returns the result.
    """
    return asyncio.run(
        scrape_and_clean(
            stay_dates,
            settings=settings,
            target_names=target_names,
            incremental=incremental,
        )
    )
