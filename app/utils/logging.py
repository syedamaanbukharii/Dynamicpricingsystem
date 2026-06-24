"""Structured logging built on Loguru.

Provides JSON or human-readable output, a contextual ``request_id`` that is
propagated through async tasks via :class:`contextvars.ContextVar`, and a helper
to bind component names (api, scraper, etl, ml, agent) to every record.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

from loguru import logger

from app.config import get_settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_CONFIGURED = False


def _inject_request_id(record: dict[str, Any]) -> bool:
    """Attach the current request id to every log record."""
    record["extra"].setdefault("request_id", request_id_ctx.get())
    record["extra"].setdefault("component", "app")
    return True


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, sqlalchemy) through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    """Configure global logging exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logger.remove()

    if settings.log_json:
        log_format = (
            '{{"ts":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}","level":"{level}",'
            '"component":"{extra[component]}","request_id":"{extra[request_id]}",'
            '"message":{message!r},"module":"{name}:{function}:{line}"}}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | req={extra[request_id]} | "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=log_format,
        filter=_inject_request_id,
        backtrace=settings.debug,
        diagnose=settings.debug,
        enqueue=True,
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    _CONFIGURED = True


def get_logger(component: str) -> logger.__class__:
    """Return a logger bound to a named component."""
    configure_logging()
    return logger.bind(component=component)


def set_request_id(request_id: str) -> None:
    """Set the request id for the current execution context."""
    request_id_ctx.set(request_id)
