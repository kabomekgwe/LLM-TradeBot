"""Unified market data models.

These models provide a consistent interface across all exchanges,
abstracting away platform-specific differences.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class OHLCV:
    """Unified OHLCV (candlestick) data structure.

    Works across all exchanges - Binance, Kraken, Coinbase, Alpaca, etc.
    """

    timestamp: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        """Validate data integrity."""
        if self.high < self.low:
            raise ValueError(f"Invalid OHLCV: high ({self.high}) < low ({self.low})")
        if self.close <= 0:
            raise ValueError(f"Invalid OHLCV: close ({self.close}) <= 0")
        if self.volume < 0:
            raise ValueError(f"Invalid OHLCV: volume ({self.volume}) < 0")

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000)

    @property
    def body(self) -> float:
        """Candle body size (absolute)."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Upper wick/shadow size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Lower wick/shadow size."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Whether candle closed higher than open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Whether candle closed lower than open."""
        return self.close < self.open

    @classmethod
    def from_ccxt(cls, data: list) -> "OHLCV":
        """Create from ccxt format [timestamp, open, high, low, close, volume].

        Args:
            data: List in ccxt format

        Returns:
            OHLCV instance

        Example:
            >>> ccxt_data = [1640000000000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5]
            >>> candle = OHLCV.from_ccxt(ccxt_data)
            >>> candle.close
            42300.0
        """
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dict with all fields
        """
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Ticker:
    """Unified ticker (current price) data structure.

    Represents the current market state for a trading pair.
    """

    symbol: str
    bid: float  # Best bid price
    ask: float  # Best ask price
    last: float  # Last traded price
    volume: float  # 24h volume
    timestamp: int  # Unix timestamp in milliseconds
    high_24h: Optional[float] = None  # 24h high (if available)
    low_24h: Optional[float] = None  # 24h low (if available)
    change_24h: Optional[float] = None  # 24h change percentage (if available)

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid price."""
        mid = (self.bid + self.ask) / 2
        return (self.spread / mid) * 100 if mid > 0 else 0

    @property
    def mid_price(self) -> float:
        """Mid-market price (average of bid and ask)."""
        return (self.bid + self.ask) / 2

    @classmethod
    def from_ccxt(cls, symbol: str, data: dict) -> "Ticker":
        """Create from ccxt ticker format.

        Args:
            symbol: Trading pair symbol
            data: ccxt ticker dict

        Returns:
            Ticker instance
        """
        return cls(
            symbol=symbol,
            bid=float(data.get('bid', 0)),
            ask=float(data.get('ask', 0)),
            last=float(data.get('last', 0)),
            volume=float(data.get('baseVolume', 0)),
            timestamp=int(data.get('timestamp', 0)),
            high_24h=float(data['high']) if data.get('high') else None,
            low_24h=float(data['low']) if data.get('low') else None,
            change_24h=float(data['percentage']) if data.get('percentage') else None,
        )


@dataclass
class OrderBook:
    """Unified order book data structure.

    Represents the current buy and sell orders in the market.
    """

    symbol: str
    bids: list[tuple[float, float]]  # [(price, size), ...] sorted highest first
    asks: list[tuple[float, float]]  # [(price, size), ...] sorted lowest first
    timestamp: int  # Unix timestamp in milliseconds

    @property
    def best_bid(self) -> Optional[tuple[float, float]]:
        """Best (highest) bid price and size."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[tuple[float, float]]:
        """Best (lowest) ask price and size."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> float:
        """Spread between best bid and best ask."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]

    @property
    def mid_price(self) -> float:
        """Mid-market price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2

    def get_depth(self, side: str, levels: int = 5) -> list[tuple[float, float]]:
        """Get order book depth for specified side.

        Args:
            side: "bid" or "ask"
            levels: Number of levels to return

        Returns:
            List of (price, size) tuples
        """
        if side == "bid":
            return self.bids[:levels]
        elif side == "ask":
            return self.asks[:levels]
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'bid' or 'ask'")

    @classmethod
    def from_ccxt(cls, symbol: str, data: dict) -> "OrderBook":
        """Create from ccxt orderbook format.

        Args:
            symbol: Trading pair symbol
            data: ccxt orderbook dict with 'bids' and 'asks'

        Returns:
            OrderBook instance
        """
        return cls(
            symbol=symbol,
            bids=[(float(price), float(amount)) for price, amount in data['bids']],
            asks=[(float(price), float(amount)) for price, amount in data['asks']],
            timestamp=int(data.get('timestamp', 0)),
        )


@dataclass
class Balance:
    """Unified account balance structure.

    Represents available funds for trading.
    """

    currency: str  # e.g., "USDT", "USD", "BTC"
    free: float  # Available balance
    used: float  # Balance in open orders/positions
    total: float  # Total balance (free + used)

    def __post_init__(self):
        """Validate balance data."""
        if self.free < 0:
            raise ValueError(f"Invalid balance: free ({self.free}) < 0")
        if self.used < 0:
            raise ValueError(f"Invalid balance: used ({self.used}) < 0")
        if abs(self.total - (self.free + self.used)) > 0.01:  # Allow small rounding errors
            raise ValueError(
                f"Invalid balance: total ({self.total}) != free ({self.free}) + used ({self.used})"
            )

    @property
    def utilization_pct(self) -> float:
        """Percentage of balance currently in use."""
        return (self.used / self.total * 100) if self.total > 0 else 0

    @classmethod
    def from_ccxt(cls, currency: str, data: dict) -> "Balance":
        """Create from ccxt balance format.

        Args:
            currency: Currency code (e.g., "USDT")
            data: ccxt balance dict for this currency

        Returns:
            Balance instance
        """
        return cls(
            currency=currency,
            free=float(data.get('free', 0)),
            used=float(data.get('used', 0)),
            total=float(data.get('total', 0)),
        )
