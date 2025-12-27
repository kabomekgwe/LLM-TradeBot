"""Sentiment features from Alternative.me Fear & Greed Index.

Fetches crypto market sentiment (0-100 scale) from Alternative.me API.
Caches historical data to prevent excessive API calls.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fear_and_greed import FearAndGreedIndex

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """Extract sentiment features from Fear & Greed Index."""

    def __init__(self):
        """Initialize sentiment feature extractor."""
        self.fgi = FearAndGreedIndex()
        self._cache: Optional[pd.DataFrame] = None
        self._cache_updated: Optional[datetime] = None

    def get_current_sentiment(self) -> Dict[str, Any]:
        """Get current fear & greed index value.

        Returns:
            Dictionary with keys: value (0-100), classification (str), timestamp

        Raises:
            Exception: If API request fails
        """
        try:
            current_data = self.fgi.get_current_data()

            return {
                'fear_greed_value': float(current_data['value']),
                'fear_greed_class': current_data['value_classification'],
                'timestamp': datetime.fromtimestamp(int(current_data['timestamp']))
            }

        except Exception as e:
            logger.error(f"Failed to fetch current sentiment: {e}")
            raise

    def get_historical_sentiment(self, days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
        """Get historical fear & greed index data with caching.

        Args:
            days: Number of days of historical data (max 365)
            force_refresh: Force refresh cache even if recent

        Returns:
            DataFrame with columns: timestamp, value, value_classification

        Raises:
            Exception: If API request fails
        """
        # Check cache freshness (refresh daily)
        if not force_refresh and self._cache is not None and self._cache_updated is not None:
            if datetime.now() - self._cache_updated < timedelta(days=1):
                logger.debug("Using cached sentiment data")
                return self._cache.copy()

        try:
            # Fetch historical data
            historical_data = self.fgi.get_last_n_days(days)

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')

            # Cache results
            self._cache = df
            self._cache_updated = datetime.now()

            logger.info(f"Fetched {len(df)} days of sentiment data")

            return df.copy()

        except Exception as e:
            logger.error(f"Failed to fetch historical sentiment: {e}")
            raise

    def align_sentiment_to_ohlcv(self, ohlcv_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Align sentiment data to OHLCV timestamps preventing look-ahead bias.

        Args:
            ohlcv_df: DataFrame with OHLCV data (must have 'timestamp' column)
            sentiment_df: DataFrame with sentiment data (timestamp, value columns)

        Returns:
            OHLCV DataFrame with sentiment features added

        Raises:
            ValueError: If required columns missing
        """
        if 'timestamp' not in ohlcv_df.columns:
            raise ValueError("OHLCV DataFrame must have 'timestamp' column")

        if 'timestamp' not in sentiment_df.columns or 'value' not in sentiment_df.columns:
            raise ValueError("Sentiment DataFrame must have 'timestamp' and 'value' columns")

        # Use merge_asof with direction='backward' to prevent look-ahead bias
        # This ensures we only use sentiment data from past or present
        result = pd.merge_asof(
            ohlcv_df.sort_values('timestamp'),
            sentiment_df[['timestamp', 'value']].sort_values('timestamp').rename(columns={'value': 'fear_greed_index'}),
            on='timestamp',
            direction='backward'  # CRITICAL: Only use past sentiment
        )

        logger.debug(f"Aligned sentiment to {len(result)} OHLCV candles")

        return result
