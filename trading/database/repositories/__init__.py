"""Database repositories for data access patterns.

Provides repository pattern for clean separation between
business logic and data access.
"""

from .trade_repository import TradeRepository

__all__ = ["TradeRepository"]
