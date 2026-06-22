"""Centralized, environment-driven application configuration.

All runtime configuration is sourced from environment variables (optionally via
a local ``.env`` file). No secrets are hard-coded. Settings are validated at
load time by Pydantic and cached so the object is constructed only once.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Environment(str, Enum):
    """Deployment environment identifiers."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # -- General ---------------------------------------------------------
    app_name: str = "dynamic-pricing-system"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True

    # -- Security --------------------------------------------------------
    api_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="Shared secret required on mutating endpoints.",
    )
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # -- PostgreSQL ------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "pricing"
    postgres_password: SecretStr = SecretStr("pricing")
    postgres_db: str = "pricing"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # -- Redis -----------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600

    # -- Anthropic / LLM (cleaning & explanation only) -------------------
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-opus-4-8"
    anthropic_max_tokens: int = 1024
    anthropic_temperature: float = 0.0
    llm_enabled: bool = True

    # -- MLflow ----------------------------------------------------------
    mlflow_tracking_uri: str = "file:./data/models/mlruns"
    mlflow_experiment: str = "dynamic-pricing"

    # -- Modeling --------------------------------------------------------
    model_dir: Path = PROJECT_ROOT / "data" / "models"
    feature_store_dir: Path = PROJECT_ROOT / "data" / "features"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    optuna_trials: int = 40
    cv_splits: int = 4
    random_seed: int = 42

    # -- Scraping --------------------------------------------------------
    scrape_targets_file: Path = PROJECT_ROOT / "app" / "scraping" / "targets.yaml"
    scrape_max_concurrency: int = 4
    scrape_request_delay_seconds: float = 1.5
    scrape_timeout_seconds: float = 30.0
    scrape_max_retries: int = 3
    scrape_user_agent: str = "DynamicPricingBot/1.0 (+https://example.com/bot)"
    scrape_respect_robots: bool = True

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Allow a comma-separated string for CORS origins."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def database_dsn(self) -> str:
        """Synchronous SQLAlchemy/psycopg connection string."""
        dsn = PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            path=self.postgres_db,
        )
        return str(dsn)

    @property
    def redis_dsn(self) -> str:
        """Redis connection string."""
        dsn = RedisDsn.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            path=str(self.redis_db),
        )
        return str(dsn)

    @property
    def is_production(self) -> bool:
        """Whether the service runs in a production-like environment."""
        return self.environment in {Environment.PRODUCTION, Environment.STAGING}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated :class:`Settings` instance."""
    return Settings()
