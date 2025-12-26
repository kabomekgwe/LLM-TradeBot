"""On-Chain Metrics Analysis - Blockchain data sentiment.

Analyzes on-chain blockchain metrics to determine market sentiment:
- Network activity (transactions, active addresses)
- Exchange flows (deposits vs withdrawals)
- Miner behavior (hash rate, miner balances)
- Whale activity (large holder movements)
- NUPL (Net Unrealized Profit/Loss)

Uses Glassnode API for comprehensive on-chain data.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp


class OnChainMetrics:
    """On-chain metrics analyzer for cryptocurrency.

    Analyzes blockchain metrics to assess market sentiment
    from network activity and participant behavior.

    Setup:
        1. Sign up at https://glassnode.com
        2. Get API key from account settings
        3. Set GLASSNODE_API_KEY environment variable

    Note:
        Free tier has limited metrics and 1-day resolution.
        Paid tiers offer more metrics and higher resolution.

    Example:
        >>> onchain = OnChainMetrics("BTC")
        >>> score = await onchain.get_sentiment_score()
        >>> print(f"On-chain sentiment: {score:.2f}")
    """

    def __init__(
        self,
        symbol: str,
        api_key: Optional[str] = None,
        lookback_days: int = 7,
    ):
        """Initialize on-chain metrics analyzer.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            api_key: Glassnode API key
            lookback_days: Days to look back for metrics
        """
        self.logger = logging.getLogger(__name__)

        self.symbol = symbol.upper()
        self.lookback_days = lookback_days

        # Glassnode API
        if api_key is None:
            import os
            api_key = os.getenv("GLASSNODE_API_KEY")

        if not api_key:
            raise ValueError("Glassnode API key not provided. Set GLASSNODE_API_KEY environment variable.")

        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"

        # Metric weights for sentiment calculation
        self.metric_weights = {
            'active_addresses': 0.2,
            'exchange_netflow': 0.25,
            'nupl': 0.25,
            'transaction_count': 0.15,
            'hash_rate': 0.15,
        }

    async def get_sentiment_score(self) -> float:
        """Get aggregated on-chain sentiment score.

        Returns:
            Sentiment score (0-1, where 0.5 is neutral)
        """
        self.logger.info(f"Fetching on-chain sentiment for {self.symbol}...")

        # Collect metrics
        metrics = await self._fetch_metrics()

        if not metrics:
            self.logger.warning("No on-chain metrics available, returning neutral sentiment")
            return 0.5

        # Calculate sentiment from metrics
        sentiment_score = self._calculate_sentiment(metrics)

        self.logger.info(f"On-chain sentiment for {self.symbol}: {sentiment_score:.2f}")

        return sentiment_score

    async def _fetch_metrics(self) -> Dict[str, float]:
        """Fetch on-chain metrics from Glassnode.

        Returns:
            Dictionary of metric values
        """
        metrics = {}

        # Calculate time range
        since_timestamp = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp())
        until_timestamp = int(datetime.now().timestamp())

        # Fetch each metric
        metric_endpoints = {
            'active_addresses': 'addresses/active_count',
            'transaction_count': 'transactions/count',
            'exchange_netflow': 'transactions/transfers_volume_exchanges_net',
            'nupl': 'indicators/net_unrealized_profit_loss',
            'hash_rate': 'mining/hash_rate_mean',
        }

        async with aiohttp.ClientSession() as session:
            for metric_name, endpoint in metric_endpoints.items():
                try:
                    value = await self._fetch_metric(
                        session,
                        endpoint,
                        since_timestamp,
                        until_timestamp
                    )
                    if value is not None:
                        metrics[metric_name] = value

                except Exception as e:
                    self.logger.error(f"Failed to fetch {metric_name}: {e}")
                    continue

        return metrics

    async def _fetch_metric(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        since: int,
        until: int,
    ) -> Optional[float]:
        """Fetch single metric from Glassnode.

        Args:
            session: aiohttp session
            endpoint: Metric endpoint
            since: Start timestamp
            until: End timestamp

        Returns:
            Metric value or None
        """
        url = f"{self.base_url}/{endpoint}"

        params = {
            'a': self.symbol,
            's': since,
            'u': until,
            'api_key': self.api_key,
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data and len(data) > 0:
                        # Get most recent value
                        latest = data[-1]
                        return float(latest['v'])

                elif response.status == 429:
                    self.logger.warning(f"Rate limited on {endpoint}")

                else:
                    error_text = await response.text()
                    self.logger.error(f"API error on {endpoint}: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Request failed for {endpoint}: {e}")

        return None

    def _calculate_sentiment(self, metrics: Dict[str, float]) -> float:
        """Calculate sentiment from on-chain metrics.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Sentiment score (0-1)
        """
        sentiment_scores = []
        weights = []

        # Active Addresses (more = bullish)
        if 'active_addresses' in metrics:
            score = self._normalize_metric(metrics['active_addresses'], 'active_addresses')
            sentiment_scores.append(score)
            weights.append(self.metric_weights['active_addresses'])

        # Exchange Net Flow (negative = bullish, coins leaving exchanges)
        if 'exchange_netflow' in metrics:
            # Negative flow is bullish (accumulation)
            netflow = metrics['exchange_netflow']
            if netflow < 0:
                score = 0.6 + (abs(netflow) / 1e9 * 0.2)  # Scale based on volume
            elif netflow > 0:
                score = 0.4 - (netflow / 1e9 * 0.2)
            else:
                score = 0.5

            score = max(0, min(1, score))  # Clamp to [0, 1]
            sentiment_scores.append(score)
            weights.append(self.metric_weights['exchange_netflow'])

        # NUPL (Net Unrealized Profit/Loss)
        if 'nupl' in metrics:
            nupl = metrics['nupl']
            # NUPL ranges from -1 to 1
            # Convert to 0-1 sentiment score
            score = (nupl + 1) / 2
            sentiment_scores.append(score)
            weights.append(self.metric_weights['nupl'])

        # Transaction Count (more = bullish)
        if 'transaction_count' in metrics:
            score = self._normalize_metric(metrics['transaction_count'], 'transaction_count')
            sentiment_scores.append(score)
            weights.append(self.metric_weights['transaction_count'])

        # Hash Rate (increasing = bullish)
        if 'hash_rate' in metrics:
            score = self._normalize_metric(metrics['hash_rate'], 'hash_rate')
            sentiment_scores.append(score)
            weights.append(self.metric_weights['hash_rate'])

        if not sentiment_scores:
            return 0.5

        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(sentiment_scores) / len(sentiment_scores)

        weighted_sentiment = sum(s * w for s, w in zip(sentiment_scores, weights)) / total_weight

        return weighted_sentiment

    def _normalize_metric(self, value: float, metric_name: str) -> float:
        """Normalize metric value to 0-1 sentiment score.

        Args:
            value: Raw metric value
            metric_name: Name of the metric

        Returns:
            Normalized score (0-1)
        """
        # Define typical ranges for each metric
        # These are approximate and should be adjusted based on current market conditions
        ranges = {
            'active_addresses': (300000, 1000000),  # BTC typical range
            'transaction_count': (200000, 400000),
            'hash_rate': (100e18, 500e18),  # Exahashes
        }

        if metric_name not in ranges:
            # Default normalization
            return 0.5

        min_val, max_val = ranges[metric_name]

        # Normalize to 0-1
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        # Clamp to [0, 1]
        return max(0, min(1, normalized))

    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed on-chain metrics.

        Returns:
            Dictionary with detailed metrics and sentiment
        """
        metrics = await self._fetch_metrics()

        sentiment_score = self._calculate_sentiment(metrics) if metrics else 0.5

        return {
            'symbol': self.symbol,
            'score': sentiment_score,
            'signal': 'bullish' if sentiment_score > 0.6 else ('bearish' if sentiment_score < 0.4 else 'neutral'),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

    def get_metric_trend(self, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """Get historical trend for a specific metric.

        Args:
            metric_name: Name of the metric
            days: Number of days to retrieve

        Returns:
            Dictionary with trend data

        Note:
            This requires additional API calls and is rate-limited.
        """
        # Placeholder for trend analysis
        self.logger.warning("Metric trend analysis requires additional implementation")
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return f"OnChainMetrics(symbol={self.symbol}, lookback_days={self.lookback_days})"
