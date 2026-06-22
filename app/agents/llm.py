"""Thin Claude (Anthropic) client used only for non-prediction tasks.

Per the system design, large language models are used **exclusively** for data
cleaning, normalization, explanation generation, and product intelligence --
never for price prediction. This wrapper centralizes model selection, secret
handling, and error translation, and is intentionally tolerant: if the LLM is
disabled, no API key is configured, or the ``anthropic`` package is unavailable,
:attr:`LLMClient.enabled` is ``False`` and callers fall back to deterministic
rule-based logic so the platform remains fully functional offline.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings, get_settings
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger

logger = get_logger("agent")


class LLMClient:
    """Synchronous Claude client for cleaning and explanation workloads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None
        self._unavailable_reason: str | None = None

    @property
    def model(self) -> str:
        """The configured Claude model identifier."""
        return self.settings.anthropic_model

    @property
    def enabled(self) -> bool:
        """Whether live LLM calls can be made (config + key + SDK present)."""
        if not self.settings.llm_enabled:
            return False
        if not self.settings.anthropic_api_key.get_secret_value():
            return False
        return self._ensure_client() is not None

    def _ensure_client(self) -> Any | None:
        """Lazily construct the Anthropic client, tolerating an absent SDK."""
        if self._client is not None:
            return self._client
        if self._unavailable_reason is not None:
            return None
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            self._unavailable_reason = f"anthropic SDK not installed: {exc}"
            logger.warning(self._unavailable_reason)
            return None
        key = self.settings.anthropic_api_key.get_secret_value()
        if not key:
            self._unavailable_reason = "no Anthropic API key configured"
            return None
        self._client = Anthropic(api_key=key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the text completion for a system+user prompt.

        Raises :class:`LLMError` if the client is unavailable or the call fails,
        allowing callers to catch it and degrade gracefully.
        """
        client = self._ensure_client()
        if client is None or not self.settings.llm_enabled:
            raise LLMError(
                "LLM is disabled or unavailable.",
                details={"reason": self._unavailable_reason or "llm_disabled"},
            )
        try:
            message = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=max_tokens or self.settings.anthropic_max_tokens,
                temperature=(
                    temperature
                    if temperature is not None
                    else self.settings.anthropic_temperature
                ),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError("Anthropic completion failed.", details={"error": str(exc)}) from exc

        text = "".join(
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMError("Anthropic returned an empty completion.")
        return text

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> Any:
        """Return parsed JSON from a completion, stripping markdown fences."""
        raw = self.complete(
            system=system + " Respond with valid JSON only, no prose or markdown.",
            user=user,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)
            cleaned = cleaned[1] if len(cleaned) > 1 else raw
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Could not parse JSON from the LLM response.",
                details={"error": str(exc), "raw": raw[:500]},
            ) from exc


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Construct an :class:`LLMClient` from settings."""
    return LLMClient(settings)
