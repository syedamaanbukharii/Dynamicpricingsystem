"""Lightweight retry/backoff helpers with no external dependency.

Supports both synchronous and asynchronous callables with exponential backoff
and optional jitter, and logs each retry attempt under the ``retry`` component.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.utils.logging import get_logger

T = TypeVar("T")
logger = get_logger("retry")


def _delay(attempt: int, base: float, factor: float, maximum: float, jitter: bool) -> float:
    """Compute the backoff delay for a given attempt (1-indexed)."""
    raw = min(maximum, base * (factor ** (attempt - 1)))
    if jitter:
        raw = raw * (0.5 + random.random() / 2.0)
    return raw


def retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator adding exponential backoff to a synchronous function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    sleep_for = _delay(attempt, base_delay, factor, max_delay, jitter)
                    logger.warning(
                        "retry {}/{} for {} after error: {} (sleep {:.2f}s)",
                        attempt,
                        attempts,
                        func.__name__,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def async_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator adding exponential backoff to an async function."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    sleep_for = _delay(attempt, base_delay, factor, max_delay, jitter)
                    logger.warning(
                        "async retry {}/{} for {} after error: {} (sleep {:.2f}s)",
                        attempt,
                        attempts,
                        func.__name__,
                        exc,
                        sleep_for,
                    )
                    await asyncio.sleep(sleep_for)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
