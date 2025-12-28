"""SQLAlchemy models for trade history.

Defines database schema for persistent trade storage with TimescaleDB
time-series optimization.
"""

from sqlalchemy import Column, String, Float, Boolean, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class TradeHistory(Base):
    """Trade history model with time-series optimization.

    Stores all trade records with full context for performance analysis.
    Optimized for time-series queries using TimescaleDB hypertable.

    Note: Composite primary key (timestamp, trade_id) is required for TimescaleDB
    hypertable partitioning.

    Example:
        >>> trade = TradeHistory(
        ...     trade_id="trade_123",
        ...     symbol="BTC/USDT",
        ...     timestamp=datetime.now(),
        ...     side="buy",
        ...     order_type="market",
        ...     amount=0.1,
        ...     entry_price=50000.0
        ... )
        >>> session.add(trade)
        >>> session.commit()
    """

    __tablename__ = 'trade_history'

    # Composite primary key (required for TimescaleDB hypertable)
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    trade_id = Column(Text, primary_key=True, nullable=False, index=True)

    # Trade identification
    symbol = Column(Text, nullable=False, index=True)

    # Trade details
    side = Column(Text, nullable=False)  # buy/sell
    order_type = Column(Text, nullable=False)  # market/limit
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)

    # Performance metrics
    realized_pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)

    # Market context
    market_regime = Column(Text, nullable=True)
    bull_confidence = Column(Float, nullable=True)
    bear_confidence = Column(Float, nullable=True)
    decision_confidence = Column(Float, nullable=True)

    # Trade outcome
    won = Column(Boolean, default=False)
    closed = Column(Boolean, default=False)
    close_timestamp = Column(TIMESTAMP(timezone=True), nullable=True)

    # Agent insights (JSONB for PostgreSQL optimization)
    agent_votes = Column(JSONB, nullable=True)
    signals = Column(JSONB, nullable=True)

    def __repr__(self):
        """String representation of trade record."""
        return (
            f"<TradeHistory(trade_id='{self.trade_id}', "
            f"symbol='{self.symbol}', "
            f"side='{self.side}', "
            f"pnl={self.realized_pnl:.2f})>"
        )

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "side": self.side,
            "order_type": self.order_type,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "pnl_pct": self.pnl_pct,
            "fees": self.fees,
            "market_regime": self.market_regime,
            "bull_confidence": self.bull_confidence,
            "bear_confidence": self.bear_confidence,
            "decision_confidence": self.decision_confidence,
            "won": self.won,
            "closed": self.closed,
            "close_timestamp": self.close_timestamp.isoformat() if self.close_timestamp else None,
            "agent_votes": self.agent_votes,
            "signals": self.signals,
        }
