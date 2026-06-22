"""FastAPI dependencies: authentication, database sessions, and services.

API-key authentication is enforced via the ``X-API-Key`` header. To keep local
development and the bundled demo frictionless, authentication is *bypassed* only
when the configured key is still the default sentinel **and** the environment is
not production; in production a correct key is always required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Header, Request

from app.config import Settings, get_settings
from app.services.pricing_service import PricingService, get_pricing_service
from app.utils.exceptions import AuthenticationError
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

logger = get_logger("api")

_DEFAULT_API_KEY = "change-me-in-production"
API_KEY_HEADER = "X-API-Key"


def settings_dependency() -> Settings:
    """Return application settings (cached singleton)."""
    return get_settings()


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    settings: Settings = Depends(settings_dependency),
) -> None:
    """Validate the ``X-API-Key`` header.

    Raises:
        AuthenticationError: If the key is missing or incorrect (and auth is not
            bypassed for local/non-production use of the default key).
    """
    configured = settings.api_key.get_secret_value()

    if configured == _DEFAULT_API_KEY and not settings.is_production:
        # Local/dev/test convenience: the operator has not set a real key.
        return

    if not x_api_key or x_api_key != configured:
        raise AuthenticationError(
            "Missing or invalid API key.",
            details={"header": API_KEY_HEADER},
        )


def get_pricing_service_dependency() -> PricingService:
    """Provide the process-wide pricing service instance."""
    return get_pricing_service()


def get_db_session(request: Request) -> Session:  # pragma: no cover - DB optional
    """Yield a database session for the duration of a request.

    This is provided for routers that need persistence. It is intentionally not
    imported at module load so that the API runs without a database driver when
    persistence is not used.
    """
    from app.database import get_db

    yield from get_db()
