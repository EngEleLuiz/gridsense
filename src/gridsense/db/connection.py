"""
TimescaleDB / PostgreSQL connection management.

Uses SQLAlchemy 2.x with a thread-safe connection pool.  The DATABASE_URL
environment variable is the single source of truth for credentials:

    postgresql://gridsense:gridsense@localhost:5432/gridsense

For tests, point DATABASE_URL at a throwaway ``gridsense_test`` database.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

_DEFAULT_URL = "postgresql://gridsense:gridsense@localhost:5432/gridsense"


def get_database_url() -> str:
    """Return the active database URL from the environment."""
    return os.environ.get("DATABASE_URL", _DEFAULT_URL)


def create_db_engine(
    database_url: str | None = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
) -> Engine:
    """Create and configure a SQLAlchemy engine.

    Parameters
    ----------
    database_url:
        Full DSN string.  Falls back to ``DATABASE_URL`` env var, then the
        local default.
    pool_size:
        Number of persistent connections in the pool.
    max_overflow:
        Extra connections allowed above ``pool_size``.
    echo:
        Log every SQL statement (useful for debugging).

    Returns
    -------
    Engine
        Configured SQLAlchemy engine.
    """
    url = database_url or get_database_url()
    return create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
        pool_pre_ping=True,  # verify connections before handing them out
    )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a scoped session factory bound to ``engine``."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------------------------------------------------------------------
# Context-manager helper
# ---------------------------------------------------------------------------


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Yield a database session, committing on success or rolling back on error.

    Usage::

        with get_session(engine) as session:
            session.add(record)

    """
    factory = make_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def ping(engine: Engine) -> bool:
    """Return ``True`` if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
