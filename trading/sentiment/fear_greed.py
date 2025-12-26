"""Fear & Greed Index - Market sentiment indicator.

Analyzes the Crypto Fear & Greed Index, which aggregates:
- Volatility (25%)
- Market momentum/volume (25%)
- Social media (15%)
- Surveys (15%)
- Dominance (10%)
- Trends (10%)

Data from Alternative.me API (free, no key required).
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp


class FearGreedIndex:
    """Fear & Greed Index analyzer.

    Fetches and analyzes the Crypto Fear & Greed Index to
    determine overall market sentiment.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed):
    - 0-24: Extreme Fear
    - 25-49: Fear
    - 50-74: Greed
    - 75-100: Extreme Greed

    Data Source: Alternative.me (free, no API key required)

    Example:
        >>> fear_greed = FearGreedIndex()
        >>> score = await fear_greed.get_sentiment_score()
        >>> print(f"Market sentiment: {score:.2f}")
    """

    def __init__(self, cache_ttl_minutes: int = 60):
        """Initialize Fear & Greed Index analyzer.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = logging.getLogger(__name__)

        self.api_url = "https://api.alternative.me/fng/"

        # Cache
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cached_value: Optional[float] = None
        self._cache_timestamp: Optional[datetime] = None

    async def get_sentiment_score(self) -> float:
        """Get Fear & Greed Index as sentiment score.

        Returns:
            Sentiment score (0-1, where 0.5 is neutral)
        """
        # Check cache
        if self._is_cache_valid():
            self.logger.debug("Returning cached Fear & Greed value")
            return self._cached_value

        self.logger.info("Fetching Fear & Greed Index...")

        try:
            # Fetch current value
            index_value = await self._fetch_current_value()

            if index_value is None:
                self.logger.warning("Failed to fetch Fear & Greed Index, returning neutral")
                return 0.5

            # Convert from 0-100 to 0-1
            normalized_score = index_value / 100

            # Cache result
            self._cached_value = normalized_score
            self._cache_timestamp = datetime.now()

            # Log with classification
            classification = self._classify_value(index_value)
            self.logger.info(f"Fear & Greed Index: {index_value:.0f} ({classification})")

            return normalized_score

        except Exception as e:
            self.logger.error(f"Failed to get Fear & Greed Index: {e}")
            return 0.5

    async def _fetch_current_value(self) -> Optional[float]:
        """Fetch current Fear & Greed Index value.

        Returns:
            Index value (0-100) or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params={'limit': 1}) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data and 'data' in data and len(data['data']) > 0:
                            value = float(data['data'][0]['value'])
                            return value

                    else:
                        error_text = await response.text()
                        self.logger.error(f"API error: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Request failed: {e}")

        return None

    async def get_historical_values(self, days: int = 30) -> Dict[str, Any]:
        """Get historical Fear & Greed Index values.

        Args:
            days: Number of days to retrieve

        Returns:
            Dictionary with historical data
        """
        self.logger.info(f"Fetching {days} days of Fear & Greed history...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params={'limit': days}) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data and 'data' in data:
                            historical = []

                            for entry in data['data']:
                                historical.append({
                                    'timestamp': datetime.fromtimestamp(int(entry['timestamp'])).isoformat(),
                                    'value': float(entry['value']),
                                    'classification': entry['value_classification'],
                                })

                            # Calculate statistics
                            values = [float(entry['value']) for entry in data['data']]
                            avg_value = sum(values) / len(values) if values else 50

                            return {
                                'period_days': days,
                                'average': avg_value,
                                'current': values[0] if values else None,
                                'min': min(values) if values else None,
                                'max': max(values) if values else None,
                                'data': historical,
                            }

        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")

        return {}

    async def get_detailed_sentiment(self) -> Dict[str, Any]:
        """Get detailed sentiment analysis with historical context.

        Returns:
            Dictionary with detailed sentiment metrics
        """
        # Get current value
        current_value = await self._fetch_current_value()

        if current_value is None:
            return {
                'score': 0.5,
                'signal': 'neutral',
                'value': 50,
                'classification': 'Unknown',
            }

        # Get historical data for trend
        historical = await self.get_historical_values(days=7)

        # Calculate trend
        trend = None
        if historical and 'data' in historical and len(historical['data']) > 1:
            week_ago = float(historical['data'][-1]['value'])
            change = current_value - week_ago
            trend = 'increasing' if change > 5 else ('decreasing' if change < -5 else 'stable')

        normalized_score = current_value / 100

        return {
            'score': normalized_score,
            'signal': self._score_to_signal(normalized_score),
            'value': current_value,
            'classification': self._classify_value(current_value),
            'trend': trend,
            'weekly_average': historical.get('average') if historical else None,
            'timestamp': datetime.now().isoformat(),
        }

    def _classify_value(self, value: float) -> str:
        """Classify Fear & Greed Index value.

        Args:
            value: Index value (0-100)

        Returns:
            Classification string
        """
        if value <= 24:
            return "Extreme Fear"
        elif value <= 49:
            return "Fear"
        elif value <= 74:
            return "Greed"
        else:
            return "Extreme Greed"

    def _score_to_signal(self, score: float) -> str:
        """Convert sentiment score to signal.

        Args:
            score: Sentiment score (0-1)

        Returns:
            Signal string: "bearish", "neutral", or "bullish"
        """
        if score < 0.3:
            return "bearish"
        elif score > 0.7:
            return "bullish"
        else:
            return "neutral"

    def _is_cache_valid(self) -> bool:
        """Check if cached value is still valid.

        Returns:
            True if cache is valid
        """
        if self._cached_value is None or self._cache_timestamp is None:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < self.cache_ttl

    def clear_cache(self):
        """Clear cached value."""
        self._cached_value = None
        self._cache_timestamp = None
        self.logger.debug("Cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        return "FearGreedIndex(source=alternative.me)"
