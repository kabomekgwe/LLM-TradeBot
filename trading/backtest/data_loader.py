"""Historical Data Loader - Fetches and caches historical OHLCV data.

Loads historical market data from exchanges or local cache for backtesting.
Supports multiple timeframes and efficient data caching.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json
import asyncio

from ..models.market_data import OHLCV
from ..providers.factory import create_provider
from ..config import TradingConfig


class HistoricalDataLoader:
    """Loads and caches historical OHLCV data for backtesting.

    Features:
    - Fetches data from exchanges via ccxt
    - Caches data locally to avoid repeated API calls
    - Supports multiple timeframes (5m, 15m, 1h, 4h, 1d)
    - Handles pagination for large date ranges

    Example:
        >>> loader = HistoricalDataLoader()
        >>> data = await loader.load_data(
        ...     symbol="BTC/USDT",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 6, 30),
        ...     timeframes=["5m", "15m", "1h"]
        ... )
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data loader.

        Args:
            cache_dir: Directory for data cache (defaults to data/historical/)
        """
        self.logger = logging.getLogger(__name__)

        # Set cache directory
        if cache_dir is None:
            cache_dir = Path("data/historical")

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Historical data cache: {self.cache_dir}")

    async def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str] = ["1h"],
        use_cache: bool = True,
    ) -> Dict[str, List[OHLCV]]:
        """Load historical data for multiple timeframes.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframes: List of timeframes to load (e.g., ["5m", "15m", "1h"])
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping timeframe to list of OHLCV candles

        Example:
            >>> data = await loader.load_data(
            ...     "BTC/USDT",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 6, 30),
            ...     ["5m", "15m", "1h"]
            ... )
            >>> len(data["1h"])
            4000
        """
        self.logger.info(
            f"Loading {symbol} data from {start_date} to {end_date} "
            f"for timeframes: {timeframes}"
        )

        result = {}

        for timeframe in timeframes:
            # Check cache first
            if use_cache:
                cached_data = self._load_from_cache(symbol, start_date, end_date, timeframe)
                if cached_data:
                    self.logger.info(f"Loaded {len(cached_data)} {timeframe} candles from cache")
                    result[timeframe] = cached_data
                    continue

            # Fetch from exchange
            self.logger.info(f"Fetching {timeframe} data from exchange...")
            data = await self._fetch_from_exchange(symbol, start_date, end_date, timeframe)

            if data:
                result[timeframe] = data

                # Save to cache
                if use_cache:
                    self._save_to_cache(symbol, start_date, end_date, timeframe, data)

        return result

    async def _fetch_from_exchange(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> List[OHLCV]:
        """Fetch historical data from exchange."""
        try:
            # Create provider (using default exchange)
            # Note: In production, make this configurable
            config = TradingConfig.from_env("binance_spot")  # Use spot for historical data
            provider = create_provider(config)

            # Calculate how many candles we need
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = int((end_date - start_date).total_seconds() / 60)
            total_candles = total_minutes // timeframe_minutes

            self.logger.info(f"Need to fetch ~{total_candles} candles")

            # Fetch in batches (most exchanges limit to 1000 candles per request)
            batch_size = 1000
            all_candles = []

            current_date = start_date
            while current_date < end_date:
                # Fetch batch
                try:
                    candles = await provider.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=batch_size,
                    )

                    if not candles:
                        break

                    # Filter to date range
                    filtered = [
                        c for c in candles
                        if start_date.timestamp() * 1000 <= c.timestamp <= end_date.timestamp() * 1000
                    ]

                    all_candles.extend(filtered)

                    # Move to next batch
                    if candles:
                        last_timestamp = candles[-1].timestamp
                        current_date = datetime.fromtimestamp(last_timestamp / 1000)
                    else:
                        break

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    self.logger.warning(f"Failed to fetch batch: {e}")
                    break

            # Close provider
            await provider.close()

            # Remove duplicates and sort
            unique_candles = {c.timestamp: c for c in all_candles}
            sorted_candles = sorted(unique_candles.values(), key=lambda c: c.timestamp)

            self.logger.info(f"Fetched {len(sorted_candles)} candles from exchange")

            return sorted_candles

        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            return []

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }

        return mapping.get(timeframe, 60)

    def _get_cache_filepath(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Path:
        """Generate cache file path."""
        # Normalize symbol for filename
        symbol_safe = symbol.replace("/", "_")

        # Create filename with date range
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        filename = f"{symbol_safe}_{timeframe}_{start_str}_to_{end_str}.json"

        return self.cache_dir / filename

    def _load_from_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Optional[List[OHLCV]]:
        """Load data from cache if available."""
        filepath = self._get_cache_filepath(symbol, start_date, end_date, timeframe)

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Convert back to OHLCV objects
            candles = [OHLCV.from_dict(candle) for candle in data]

            self.logger.debug(f"Loaded {len(candles)} candles from cache: {filepath.name}")

            return candles

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        candles: List[OHLCV],
    ):
        """Save data to cache."""
        filepath = self._get_cache_filepath(symbol, start_date, end_date, timeframe)

        try:
            # Convert to dictionaries
            data = [candle.to_dict() for candle in candles]

            with open(filepath, "w") as f:
                json.dump(data, f)

            self.logger.debug(f"Saved {len(candles)} candles to cache: {filepath.name}")

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data.

        Args:
            symbol: If provided, only clear cache for this symbol.
                   If None, clear entire cache.
        """
        if symbol is None:
            # Clear all cache files
            for filepath in self.cache_dir.glob("*.json"):
                filepath.unlink()
            self.logger.info("Cleared entire cache")
        else:
            # Clear cache for specific symbol
            symbol_safe = symbol.replace("/", "_")
            for filepath in self.cache_dir.glob(f"{symbol_safe}_*.json"):
                filepath.unlink()
            self.logger.info(f"Cleared cache for {symbol}")

    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached data.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": [f.name for f in cache_files],
        }
