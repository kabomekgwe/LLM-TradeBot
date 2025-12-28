"""Database connection and session management.

Provides SQLAlchemy engine and session factory for database operations.
Supports connection pooling and automatic session cleanup.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://tradingbot:changeme@localhost:5437/tradingbot'
)

# Create engine with connection pooling
# pool_pre_ping: Check connection health before using
# pool_size: Number of connections to maintain
# max_overflow: Additional connections allowed during peak load
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator:
    """Get database session with automatic cleanup.

    Use as dependency injection in FastAPI or context manager.

    Example (FastAPI):
        >>> from fastapi import Depends
        >>> @app.get("/trades")
        ... def get_trades(db: Session = Depends(get_db)):
        ...     return db.query(TradeHistory).all()

    Example (Context Manager):
        >>> from contextlib import contextmanager
        >>> with next(get_db()) as db:
        ...     trades = db.query(TradeHistory).all()

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables.

    Creates all tables defined in models if they don't exist.
    For production, use Alembic migrations instead.

    Example:
        >>> from trading.database.connection import init_db
        >>> from trading.database.models import Base
        >>> Base.metadata.create_all(bind=engine)
    """
    from .models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


def test_connection() -> bool:
    """Test database connection.

    Returns:
        True if connection successful, False otherwise

    Example:
        >>> from trading.database.connection import test_connection
        >>> if test_connection():
        ...     print("Database connected!")
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
