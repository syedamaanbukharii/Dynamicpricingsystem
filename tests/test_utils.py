"""Tests for utility helpers: retry backoff and exception serialization."""

from __future__ import annotations

import pytest
from app.utils.exceptions import (
    AuthenticationError,
    PricingError,
    PricingSystemError,
    ValidationError,
)
from app.utils.retry import retry


def test_retry_succeeds_after_failures() -> None:
    """The retry decorator retries until the call succeeds."""
    calls = {"n": 0}

    @retry(attempts=4, base_delay=0.0, jitter=False, exceptions=(ValueError,))
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhaustion() -> None:
    """The retry decorator re-raises the last error after exhausting attempts."""
    calls = {"n": 0}

    @retry(attempts=2, base_delay=0.0, jitter=False, exceptions=(ValueError,))
    def always_fail() -> None:
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fail()
    assert calls["n"] == 2


def test_retry_does_not_catch_unlisted_exceptions() -> None:
    """Exceptions not in the configured tuple propagate immediately."""
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.0, jitter=False, exceptions=(ValueError,))
    def raises_type_error() -> None:
        calls["n"] += 1
        raise TypeError("immediate")

    with pytest.raises(TypeError):
        raises_type_error()
    assert calls["n"] == 1


def test_exception_http_status_and_code() -> None:
    """Domain exceptions carry their HTTP status and stable error code."""
    assert ValidationError("x").http_status == 422
    assert PricingError("x").http_status == 422
    assert AuthenticationError("x").http_status == 401


def test_exception_to_dict_roundtrip() -> None:
    """Serialized errors include code, message, and details."""
    err = PricingError("bad price", details={"low": 1, "high": 2})
    payload = err.to_dict()
    assert payload["message"] == "bad price"
    assert payload["details"] == {"low": 1, "high": 2}
    assert isinstance(err, PricingSystemError)
