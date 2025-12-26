"""DataSyncAgent - Fetches multi-timeframe market data.

This is Agent #1 in the 8-agent adversarial decision system.
Responsible for asynchronously fetching market data across multiple timeframes.
"""

import asyncio
from typing import Any

from .base_agent import BaseAgent


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

        self.log_decision(f"Fetching market data for {symbol}")

        # Fetch multiple timeframes in parallel for efficiency
        timeframes = ["5m", "15m", "1h"]

        # Create tasks for parallel execution
        tasks = [
            self.provider.fetch_ohlcv(symbol, tf, limit=200)
            for tf in timeframes
        ]

        # Also fetch ticker and orderbook
        tasks.append(self.provider.fetch_ticker(symbol))
        tasks.append(self.provider.fetch_orderbook(symbol, limit=20))

        try:
            # Execute all fetches in parallel
            results = await asyncio.gather(*tasks)

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
                f"Successfully fetched data: "
                f"5m={len(ohlcv_5m)}, 15m={len(ohlcv_15m)}, 1h={len(ohlcv_1h)} candles"
            )

            return {"market_data": market_data}

        except Exception as e:
            self.log_decision(f"Failed to fetch market data: {e}", level="error")
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
