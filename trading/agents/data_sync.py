"""DataSyncAgent - Fetches multi-timeframe market data.

This is Agent #1 in the 8-agent adversarial decision system.
Responsible for asynchronously fetching market data across multiple timeframes.
"""

import asyncio
from typing import Any

import ccxt

from .base_agent import BaseAgent
from ..logging_config import DecisionContext
from ..exceptions import (
    ExchangeConnectionError,
    RateLimitExceededError,
    InvalidSymbolError,
    AgentTimeoutError,
    TradingBotError,
)


class DataSyncAgent(BaseAgent):
    """Data synchronization agent.

    Fetches multi-timeframe OHLCV data, ticker, and orderbook data
    in parallel for efficient data gathering.
    """

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fetch multi-timeframe market data.

        Args:
            context: Must contain "symbol" key (e.g., "BTC/USDT")

        Returns:
            Context updated with "market_data" containing:
                - "5m", "15m", "1h": OHLCV lists for each timeframe
                - "ticker": Current ticker data
                - "orderbook": Current order book

        Example:
            >>> context = {"symbol": "BTC/USDT"}
            >>> result = await agent.execute(context)
            >>> result["market_data"]["1h"][0].close
            42000.0
        """
        symbol = context.get("symbol")
        if not symbol:
            raise ValueError("Symbol is required in context")

        self.log_decision(
            "fetching_market_data",
            symbol=symbol
        )

        # Fetch multiple timeframes in parallel for efficiency
        timeframes = ["5m", "15m", "1h"]

        try:
            # Execute fetches with timeout protection (30s each for OHLCV)
            # Use asyncio.wait_for for individual timeout control
            ohlcv_5m = await asyncio.wait_for(
                self.provider.fetch_ohlcv(symbol, "5m", limit=200),
                timeout=30.0
            )
            ohlcv_15m = await asyncio.wait_for(
                self.provider.fetch_ohlcv(symbol, "15m", limit=200),
                timeout=30.0
            )
            ohlcv_1h = await asyncio.wait_for(
                self.provider.fetch_ohlcv(symbol, "1h", limit=200),
                timeout=30.0
            )

            # Fetch ticker and orderbook with shorter timeout (10s)
            ticker = await asyncio.wait_for(
                self.provider.fetch_ticker(symbol),
                timeout=10.0
            )
            orderbook = await asyncio.wait_for(
                self.provider.fetch_orderbook(symbol, limit=20),
                timeout=10.0
            )

            # Unpack for compatibility with existing code
            results = [ohlcv_5m, ohlcv_15m, ohlcv_1h, ticker, orderbook]

            # Unpack results
            ohlcv_5m, ohlcv_15m, ohlcv_1h, ticker, orderbook = results

            # Validate data integrity
            if not self._validate_ohlcv_data(ohlcv_1h):
                self.log_decision("Invalid OHLCV data detected", level="warning")

            market_data = {
                "5m": ohlcv_5m,
                "15m": ohlcv_15m,
                "1h": ohlcv_1h,
                "ticker": ticker,
                "orderbook": orderbook,
            }

            self.log_decision(
                "data_sync_complete",
                symbol=symbol,
                candles_5m=len(ohlcv_5m),
                candles_15m=len(ohlcv_15m),
                candles_1h=len(ohlcv_1h),
                ticker_price=ticker.last if ticker else None,
            )

            return {"market_data": market_data}

        except asyncio.TimeoutError:
            # Timeout from asyncio.wait_for - convert to our exception
            self.log_decision(
                "data_fetch_timeout",
                level="error",
                symbol=symbol,
                timeout_seconds=30,
            )
            raise AgentTimeoutError(f"Timeout fetching market data for {symbol}")
        except ccxt.NetworkError as e:
            # Network errors - retriable
            self.log_decision(
                "exchange_network_error",
                level="error",
                symbol=symbol,
                error=str(e),
            )
            raise ExchangeConnectionError(f"Network error fetching {symbol}: {e}")
        except ccxt.RateLimitExceeded as e:
            # Rate limit - should back off
            self.log_decision(
                "rate_limit_exceeded",
                level="warning",
                symbol=symbol,
                retry_after=getattr(e, "retry_after", None),
            )
            raise RateLimitExceededError(f"Rate limit exceeded for {symbol}")
        except ccxt.BadSymbol as e:
            # Invalid symbol - don't retry
            self.log_decision(
                "invalid_symbol",
                level="error",
                symbol=symbol,
            )
            raise InvalidSymbolError(f"Symbol {symbol} not supported: {e}")
        except TradingBotError:
            # Our own exceptions - re-raise
            raise
        except Exception as e:
            # Truly unexpected errors - log with full traceback
            self.log_decision(
                "unexpected_data_sync_error",
                level="critical",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    def _validate_ohlcv_data(self, ohlcv_list: list) -> bool:
        """Validate OHLCV data integrity.

        Args:
            ohlcv_list: List of OHLCV objects to validate

        Returns:
            True if data is valid, False otherwise
        """
        if not ohlcv_list:
            return False

        for candle in ohlcv_list:
            # Check basic constraints
            if candle.high < candle.low:
                self.logger.error(f"Invalid candle: high < low")
                return False

            if candle.close <= 0:
                self.logger.error(f"Invalid candle: close <= 0")
                return False

            if candle.volume < 0:
                self.logger.error(f"Invalid candle: volume < 0")
                return False

        return True
