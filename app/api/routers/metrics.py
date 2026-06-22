"""Prometheus metrics exposition endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

from app.monitoring import render_latest

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", summary="Prometheus metrics exposition")
def metrics() -> Response:
    """Return the current metrics in Prometheus text exposition format."""
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)
