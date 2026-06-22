"""Configurable scrape targets and selector abstraction.

Selectors are **never hardcoded** in scraper logic; each competitor source is
described declaratively in a YAML file (see ``targets.yaml``) and loaded into
typed :class:`ScrapeTarget` objects. This keeps site-specific CSS out of the
code, lets operators add or fix targets without a deploy, and makes the parsing
layer trivially testable.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl

from app.schemas.common import Currency
from app.utils.exceptions import ConfigurationError


class ScrapeSelectors(BaseModel):
    """CSS selectors describing how to extract fields from a listing page."""

    listing: str = Field(description="Selector for each room/offer container.")
    room_name: str = Field(description="Selector for the room name within a listing.")
    price: str = Field(description="Selector for the price within a listing.")
    availability: str | None = Field(
        default=None, description="Optional selector indicating availability."
    )


class ScrapeTarget(BaseModel):
    """A single competitor source and how to crawl/parse it."""

    name: str
    competitor: str
    base_url: HttpUrl
    search_url_template: str = Field(
        description="URL template with a ``{stay_date}`` placeholder (ISO date)."
    )
    currency: Currency = Currency.USD
    selectors: ScrapeSelectors
    enabled: bool = True
    requires_javascript: bool = Field(
        default=True,
        description="If true, prefer the Playwright renderer over a plain fetch.",
    )

    def url_for(self, stay_date: str) -> str:
        """Render the search URL for a given ISO ``stay_date``."""
        try:
            return self.search_url_template.format(stay_date=stay_date)
        except (KeyError, IndexError) as exc:  # pragma: no cover - config error
            raise ConfigurationError(
                "search_url_template must contain a '{stay_date}' placeholder.",
                details={"target": self.name, "error": str(exc)},
            ) from exc


class ScrapeConfig(BaseModel):
    """Top-level scrape configuration: a collection of targets."""

    targets: list[ScrapeTarget] = Field(default_factory=list)

    def enabled_targets(self) -> list[ScrapeTarget]:
        """Return only the targets flagged as enabled."""
        return [t for t in self.targets if t.enabled]


def load_targets(path: Path) -> ScrapeConfig:
    """Load and validate the scrape configuration from a YAML file."""
    if not path.exists():
        raise ConfigurationError(
            "Scrape targets file not found.", details={"path": str(path)}
        )
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            "Could not parse scrape targets YAML.", details={"error": str(exc)}
        ) from exc
    if not isinstance(data, dict) or "targets" not in data:
        raise ConfigurationError(
            "Scrape targets YAML must contain a top-level 'targets' list.",
            details={"path": str(path)},
        )
    return ScrapeConfig.model_validate(data)
