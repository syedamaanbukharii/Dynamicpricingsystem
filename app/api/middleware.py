"""HTTP middleware: request correlation IDs, access logging, and metrics.

Each request is assigned (or inherits, via the ``X-Request-ID`` header) a unique
correlation id that is bound to the logging context and echoed back on the
response. Request counts and latencies are recorded to Prometheus.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.monitoring import record_request
from app.utils.logging import get_logger, set_request_id

logger = get_logger("api")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log access, and record request metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Bind a request id, time the request, record metrics, and log access."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)
        request.state.request_id = request_id

        endpoint = request.url.path
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled errors are logged here; the app's exception handlers
            # produce the actual client response. Record the metric and re-raise.
            duration = time.perf_counter() - start
            record_request(request.method, endpoint, 500, duration)
            logger.exception(
                "{} {} -> 500 ({:.1f} ms)",
                request.method,
                request.url.path,
                duration * 1000.0,
            )
            raise

        duration = time.perf_counter() - start
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            endpoint = route.path
        record_request(request.method, endpoint, response.status_code, duration)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "{} {} -> {} ({:.1f} ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration * 1000.0,
        )
        return response


__all__ = ["REQUEST_ID_HEADER", "JSONResponse", "RequestContextMiddleware"]
