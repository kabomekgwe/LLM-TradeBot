"""Trade repository for database operations.

Implements repository pattern for trade history CRUD operations.
Provides clean abstraction over SQLAlchemy for trade data access.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models import TradeHistory


class TradeRepository:
    """Repository for trade history operations.

    Encapsulates all database operations for trade records,
    providing a clean interface for the trading system.

    Example:
        >>> from trading.database.connection import get_db
        >>> db = next(get_db())
        >>> repo = TradeRepository(db)
        >>> trades = repo.get_recent_trades(limit=10)
    """

    def __init__(self, db: Session):
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def create_trade(self, trade_data: dict) -> TradeHistory:
        """Insert new trade record.

        Args:
            trade_data: Dictionary with trade fields

        Returns:
            Created trade record

        Example:
            >>> trade_data = {
            ...     "trade_id": "trade_123",
            ...     "symbol": "BTC/USDT",
            ...     "timestamp": datetime.now(),
            ...     "side": "buy",
            ...     "order_type": "market",
            ...     "amount": 0.1,
            ...     "entry_price": 50000.0
            ... }
            >>> trade = repo.create_trade(trade_data)
        """
        trade = TradeHistory(**trade_data)
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        return trade

    def update_trade(self, trade_id: str, updates: dict) -> Optional[TradeHistory]:
        """Update existing trade record.

        Typically used to close a position by updating exit_price,
        realized_pnl, and other outcome fields.

        Args:
            trade_id: Trade ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated trade record or None if not found

        Example:
            >>> updates = {
            ...     "exit_price": 51000.0,
            ...     "realized_pnl": 100.0,
            ...     "closed": True,
            ...     "won": True,
            ...     "close_timestamp": datetime.now()
            ... }
            >>> trade = repo.update_trade("trade_123", updates)
        """
        trade = self.db.query(TradeHistory).filter_by(trade_id=trade_id).first()
        if trade:
            for key, value in updates.items():
                setattr(trade, key, value)
            self.db.commit()
            self.db.refresh(trade)
        return trade

    def get_trade(self, trade_id: str) -> Optional[TradeHistory]:
        """Get single trade by ID.

        Args:
            trade_id: Trade ID to lookup

        Returns:
            Trade record or None if not found
        """
        return self.db.query(TradeHistory).filter_by(trade_id=trade_id).first()

    def get_recent_trades(self, limit: int = 100) -> List[TradeHistory]:
        """Get recent trades ordered by timestamp.

        Args:
            limit: Maximum number of trades to return (default 100)

        Returns:
            List of trade records, newest first

        Example:
            >>> recent_trades = repo.get_recent_trades(limit=50)
            >>> for trade in recent_trades:
            ...     print(f"{trade.symbol}: {trade.realized_pnl}")
        """
        return (
            self.db.query(TradeHistory)
            .order_by(desc(TradeHistory.timestamp))
            .limit(limit)
            .all()
        )

    def get_trades_by_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TradeHistory]:
        """Get trades for specific symbol with optional date filtering.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of trade records for the symbol

        Example:
            >>> from datetime import datetime, timedelta
            >>> start = datetime.now() - timedelta(days=7)
            >>> btc_trades = repo.get_trades_by_symbol("BTC/USDT", start_date=start)
        """
        query = self.db.query(TradeHistory).filter_by(symbol=symbol)

        if start_date:
            query = query.filter(TradeHistory.timestamp >= start_date)
        if end_date:
            query = query.filter(TradeHistory.timestamp <= end_date)

        return query.order_by(desc(TradeHistory.timestamp)).all()

    def get_closed_trades(self, limit: Optional[int] = None) -> List[TradeHistory]:
        """Get all closed trades (positions that have been exited).

        Args:
            limit: Optional limit on number of trades

        Returns:
            List of closed trade records

        Example:
            >>> closed = repo.get_closed_trades(limit=100)
            >>> total_pnl = sum(t.realized_pnl for t in closed)
        """
        query = (
            self.db.query(TradeHistory)
            .filter_by(closed=True)
            .order_by(desc(TradeHistory.close_timestamp))
        )

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_open_trades(self) -> List[TradeHistory]:
        """Get all currently open trades (positions not yet exited).

        Returns:
            List of open trade records

        Example:
            >>> open_positions = repo.get_open_trades()
            >>> print(f"Open positions: {len(open_positions)}")
        """
        return (
            self.db.query(TradeHistory)
            .filter_by(closed=False)
            .order_by(desc(TradeHistory.timestamp))
            .all()
        )

    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics from all trades.

        Returns:
            Dictionary with performance metrics:
            - total_trades: Total number of trades
            - closed_trades: Number of closed trades
            - open_trades: Number of open trades
            - winning_trades: Number of winning trades
            - losing_trades: Number of losing trades
            - win_rate: Win rate percentage
            - total_pnl: Total realized P&L
            - avg_win: Average winning trade P&L
            - avg_loss: Average losing trade P&L

        Example:
            >>> metrics = repo.get_performance_metrics()
            >>> print(f"Win Rate: {metrics['win_rate']:.1f}%")
            >>> print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        """
        all_trades = self.db.query(TradeHistory).all()
        closed_trades = [t for t in all_trades if t.closed]
        open_trades = [t for t in all_trades if not t.closed]
        winning_trades = [t for t in closed_trades if t.won]
        losing_trades = [t for t in closed_trades if not t.won]

        total_pnl = sum(t.realized_pnl for t in closed_trades)
        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))

        return {
            "total_trades": len(all_trades),
            "closed_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0,
            "total_pnl": total_pnl,
            "avg_win": (total_wins / len(winning_trades)) if winning_trades else 0.0,
            "avg_loss": (total_losses / len(losing_trades)) if losing_trades else 0.0,
            "profit_factor": (total_wins / total_losses) if total_losses > 0 else 0.0,
        }
