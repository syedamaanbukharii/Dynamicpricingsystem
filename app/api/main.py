"""FastAPI application factory.

Wires together middleware, routers, CORS, exception handling, and a lifespan
that configures logging and warms the pricing service. Domain exceptions
(:class:`~app.utils.exceptions.PricingSystemError`) are translated into a
uniform :class:`~app.schemas.common.ErrorResponse` carrying the request id.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app import __version__
from app.api.middleware import RequestContextMiddleware
from app.api.routers import (
    etl,
    explanation,
    health,
    metrics,
    prediction,
    scraping,
    training,
)
from app.config import get_settings
from app.schemas.common import ErrorResponse
from app.services.pricing_service import get_pricing_service
from app.utils.exceptions import PricingSystemError
from app.utils.logging import configure_logging, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle.

    On startup: configure logging and eagerly construct the pricing service so
    the model (or heuristic fallback) is loaded before the first request.
    """
    settings = get_settings()
    configure_logging()
    logger.info(
        "starting {} v{} (env={})",
        settings.app_name,
        __version__,
        settings.environment.value,
    )
    # Warm the pricing service (loads model or heuristic fallback).
    service = get_pricing_service()
    logger.info("pricing service warm (model_version={})", service.model_version)
    yield
    logger.info("shutting down")


def _request_id_of(request: Request) -> str | None:
    """Best-effort extraction of the correlation id bound by middleware."""
    return getattr(request.state, "request_id", None)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Recommends revenue-maximizing nightly hotel room prices using an "
            "XGBoost demand model and a rule-aware pricing engine. LLMs are used "
            "only for data cleaning, normalization, and explanations."
        ),
        lifespan=lifespan,
    )

    # CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Request id + access logging + metrics.
    app.add_middleware(RequestContextMiddleware)

    # Exception handlers -> uniform ErrorResponse.
    @app.exception_handler(PricingSystemError)
    async def _handle_domain_error(request: Request, exc: PricingSystemError) -> JSONResponse:
        body = ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=_request_id_of(request),
        )
        if exc.http_status >= 500:
            logger.error("domain error {}: {}", exc.code, exc.message)
        else:
            logger.warning("domain error {}: {}", exc.code, exc.message)
        return JSONResponse(status_code=exc.http_status, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            code="validation_error",
            message="Request validation failed.",
            details={"errors": exc.errors()},
            request_id=_request_id_of(request),
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: {}", exc)
        body = ErrorResponse(
            code="internal_error",
            message="An unexpected error occurred.",
            request_id=_request_id_of(request),
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    # Routers.
    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(prediction.router, prefix=prefix)
    app.include_router(explanation.router, prefix=prefix)
    app.include_router(scraping.router, prefix=prefix)
    app.include_router(etl.router, prefix=prefix)
    app.include_router(training.router, prefix=prefix)
    # Metrics is conventionally exposed at the root, unprefixed.
    app.include_router(metrics.router)

    logger.info("application configured with prefix {}", prefix)
    return app


app = create_app()
