"""Database engine and session management.

The engine and session factory are created lazily and cached so that merely
importing this module (for example, during unit tests or when running the
pricing service against a heuristic model) never requires a live PostgreSQL
connection or the ``psycopg`` driver to be installed. The engine is only
constructed the first time :func:`get_engine` is called.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import Settings, get_settings
from app.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session, sessionmaker

logger = get_logger("api")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) and return the SQLAlchemy engine.

    Returns:
        A process-wide singleton :class:`sqlalchemy.Engine` configured from
        :class:`~app.config.Settings`.
    """
    from sqlalchemy import create_engine

    settings: Settings = get_settings()
    dsn = settings.database_dsn
    engine = create_engine(
        dsn,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        echo=settings.db_echo,
        future=True,
    )
    logger.info(
        "Database engine created host={} db={} pool_size={}",
        settings.postgres_host,
        settings.postgres_db,
        settings.db_pool_size,
    )
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Create (once) and return a configured session factory.

    Returns:
        A :class:`sqlalchemy.orm.sessionmaker` bound to the engine. Sessions are
        configured with ``expire_on_commit=False`` so ORM objects remain usable
        after the surrounding transaction commits.
    """
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    The session is committed on success and rolled back on any exception, then
    always closed. Use this for scripts, ETL tasks, and services.

    Yields:
        An active :class:`sqlalchemy.orm.Session`.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session.

    Unlike :func:`get_session`, this does not auto-commit; routers are expected
    to commit explicitly when they mutate state. The session is always closed
    when the request finishes.

    Yields:
        An active :class:`sqlalchemy.orm.Session`.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined on the declarative metadata.

    This is convenient for local development and tests. Production deployments
    should prefer a migration tool, but ``create_all`` is idempotent and safe to
    call on an existing schema.
    """
    from app.database.models import Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured ({} tables)", len(Base.metadata.tables))


def dispose_engine() -> None:
    """Dispose the cached engine and clear the singletons.

    Primarily useful in tests to reset global state between cases.
    """
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
