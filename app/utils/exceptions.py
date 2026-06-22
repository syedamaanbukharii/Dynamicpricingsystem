"""Domain-specific exception hierarchy.

A single base class (:class:`PricingSystemError`) lets the API exception
middleware translate failures into consistent HTTP responses while preserving a
machine-readable ``code`` and an HTTP status hint.
"""

from __future__ import annotations

from typing import Any


class PricingSystemError(Exception):
    """Base class for all application errors."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for an API response body."""
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(PricingSystemError):
    """Raised when configuration is missing or invalid."""

    code = "configuration_error"
    http_status = 500


class ValidationError(PricingSystemError):
    """Raised when input data fails domain validation."""

    code = "validation_error"
    http_status = 422


class DataQualityError(PricingSystemError):
    """Raised when ingested data violates quality expectations."""

    code = "data_quality_error"
    http_status = 422


class ScrapingError(PricingSystemError):
    """Raised when a scraping operation fails irrecoverably."""

    code = "scraping_error"
    http_status = 502


class ETLError(PricingSystemError):
    """Raised when an ETL stage fails."""

    code = "etl_error"
    http_status = 500


class ModelNotFoundError(PricingSystemError):
    """Raised when no trained model is available for inference."""

    code = "model_not_found"
    http_status = 409


class TrainingError(PricingSystemError):
    """Raised when model training fails."""

    code = "training_error"
    http_status = 500


class PricingError(PricingSystemError):
    """Raised when a price cannot be computed under the supplied constraints."""

    code = "pricing_error"
    http_status = 422


class LLMError(PricingSystemError):
    """Raised when an LLM call fails or returns unusable output."""

    code = "llm_error"
    http_status = 502


class AuthenticationError(PricingSystemError):
    """Raised when an API key is missing or invalid."""

    code = "authentication_error"
    http_status = 401
