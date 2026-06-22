"""API package: FastAPI application factory and ASGI app."""

from app.api.main import app, create_app

__all__ = ["app", "create_app"]
