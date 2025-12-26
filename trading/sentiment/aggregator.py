"""Sentiment Aggregator - Combine multi-source sentiment analysis.

Aggregates sentiment from:
- Twitter/X social media
- Financial news
- On-chain blockchain metrics
- Fear & Greed Index

Provides weighted composite sentiment score.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .twitter import TwitterSentiment
from .news import NewsSentiment
from .onchain import OnChainMetrics
from .fear_greed import FearGreedIndex


class SentimentAggregator:
    """Aggregate sentiment from multiple sources.

    Combines sentiment signals from social media, news, on-chain data,
    and market indicators into a single composite score.

    Example:
        >>> aggregator = SentimentAggregator("BTC")
        >>> sentiment = await aggregator.get_sentiment()
        >>> print(f"Overall: {sentiment['overall_score']:.2f}")
    """

    def __init__(
        self,
        symbol: str,
        enable_twitter: bool = True,
        enable_news: bool = True,
        enable_onchain: bool = True,
        enable_fear_greed: bool = True,
        twitter_weight: float = 0.25,
        news_weight: float = 0.35,
        onchain_weight: float = 0.20,
        fear_greed_weight: float = 0.20,
        cache_ttl_minutes: int = 15,
        **kwargs,
    ):
        """Initialize sentiment aggregator.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            enable_twitter: Enable Twitter sentiment
            enable_news: Enable news sentiment
            enable_onchain: Enable on-chain metrics
            enable_fear_greed: Enable Fear & Greed Index
            twitter_weight: Weight for Twitter sentiment
            news_weight: Weight for news sentiment
            onchain_weight: Weight for on-chain metrics
            fear_greed_weight: Weight for Fear & Greed
            cache_ttl_minutes: Cache time-to-live in minutes
            **kwargs: Additional parameters for individual sources
        """
        self.logger = logging.getLogger(__name__)

        self.symbol = symbol.upper().replace("/", "").replace("USDT", "").replace("USD", "")

        # Source enablement
        self.enable_twitter = enable_twitter
        self.enable_news = enable_news
        self.enable_onchain = enable_onchain
        self.enable_fear_greed = enable_fear_greed

        # Weights (will be normalized)
        weights = {
            'twitter': twitter_weight if enable_twitter else 0,
            'news': news_weight if enable_news else 0,
            'onchain': onchain_weight if enable_onchain else 0,
            'fear_greed': fear_greed_weight if enable_fear_greed else 0,
        }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = weights

        # Initialize sources
        self.twitter = TwitterSentiment(self.symbol) if enable_twitter else None
        self.news = NewsSentiment(self.symbol) if enable_news else None
        self.onchain = OnChainMetrics(self.symbol) if enable_onchain else None
        self.fear_greed = FearGreedIndex() if enable_fear_greed else None

        # Cache
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cached_sentiment: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None

    async def get_sentiment(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get aggregated sentiment from all sources.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary with sentiment scores and metadata
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            self.logger.debug("Returning cached sentiment")
            return self._cached_sentiment

        self.logger.info(f"Fetching sentiment for {self.symbol}...")

        # Collect sentiment from all sources (in parallel)
        tasks = []
        if self.twitter:
            tasks.append(self._get_twitter_sentiment())
        if self.news:
            tasks.append(self._get_news_sentiment())
        if self.onchain:
            tasks.append(self._get_onchain_sentiment())
        if self.fear_greed:
            tasks.append(self._get_fear_greed_sentiment())

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract scores
        twitter_score = None
        news_score = None
        onchain_score = None
        fear_greed_score = None

        idx = 0
        if self.twitter:
            twitter_score = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1
        if self.news:
            news_score = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1
        if self.onchain:
            onchain_score = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1
        if self.fear_greed:
            fear_greed_score = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1

        # Calculate weighted average
        overall_score = self._calculate_overall_score(
            twitter_score, news_score, onchain_score, fear_greed_score
        )

        # Build result
        sentiment_result = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'overall_signal': self._score_to_signal(overall_score),
            'sources': {
                'twitter': {
                    'score': twitter_score,
                    'signal': self._score_to_signal(twitter_score) if twitter_score is not None else None,
                    'weight': self.weights.get('twitter', 0),
                    'enabled': self.enable_twitter,
                },
                'news': {
                    'score': news_score,
                    'signal': self._score_to_signal(news_score) if news_score is not None else None,
                    'weight': self.weights.get('news', 0),
                    'enabled': self.enable_news,
                },
                'onchain': {
                    'score': onchain_score,
                    'signal': self._score_to_signal(onchain_score) if onchain_score is not None else None,
                    'weight': self.weights.get('onchain', 0),
                    'enabled': self.enable_onchain,
                },
                'fear_greed': {
                    'score': fear_greed_score,
                    'signal': self._score_to_signal(fear_greed_score) if fear_greed_score is not None else None,
                    'weight': self.weights.get('fear_greed', 0),
                    'enabled': self.enable_fear_greed,
                },
            },
        }

        # Cache result
        self._cached_sentiment = sentiment_result
        self._cache_timestamp = datetime.now()

        self.logger.info(f"Sentiment aggregated: {overall_score:.2f} ({sentiment_result['overall_signal']})")

        return sentiment_result

    async def _get_twitter_sentiment(self) -> Optional[float]:
        """Get Twitter sentiment score."""
        try:
            return await self.twitter.get_sentiment_score()
        except Exception as e:
            self.logger.error(f"Failed to get Twitter sentiment: {e}")
            return None

    async def _get_news_sentiment(self) -> Optional[float]:
        """Get news sentiment score."""
        try:
            return await self.news.get_sentiment_score()
        except Exception as e:
            self.logger.error(f"Failed to get news sentiment: {e}")
            return None

    async def _get_onchain_sentiment(self) -> Optional[float]:
        """Get on-chain sentiment score."""
        try:
            return await self.onchain.get_sentiment_score()
        except Exception as e:
            self.logger.error(f"Failed to get on-chain sentiment: {e}")
            return None

    async def _get_fear_greed_sentiment(self) -> Optional[float]:
        """Get Fear & Greed Index score."""
        try:
            return await self.fear_greed.get_sentiment_score()
        except Exception as e:
            self.logger.error(f"Failed to get Fear & Greed sentiment: {e}")
            return None

    def _calculate_overall_score(
        self,
        twitter_score: Optional[float],
        news_score: Optional[float],
        onchain_score: Optional[float],
        fear_greed_score: Optional[float],
    ) -> float:
        """Calculate weighted overall sentiment score.

        Args:
            twitter_score: Twitter sentiment (0-1 or None)
            news_score: News sentiment (0-1 or None)
            onchain_score: On-chain sentiment (0-1 or None)
            fear_greed_score: Fear & Greed score (0-1 or None)

        Returns:
            Overall sentiment score (0-1)
        """
        scores = {
            'twitter': twitter_score,
            'news': news_score,
            'onchain': onchain_score,
            'fear_greed': fear_greed_score,
        }

        # Filter out None values
        valid_scores = {k: v for k, v in scores.items() if v is not None}

        if not valid_scores:
            # No valid scores, return neutral
            return 0.5

        # Calculate weighted average (renormalize weights for available sources)
        total_weight = sum(self.weights.get(k, 0) for k in valid_scores.keys())

        if total_weight == 0:
            # Equal weighting if all weights are zero
            return sum(valid_scores.values()) / len(valid_scores)

        weighted_sum = sum(
            score * self.weights.get(source, 0)
            for source, score in valid_scores.items()
        )

        return weighted_sum / total_weight

    def _score_to_signal(self, score: Optional[float]) -> Optional[str]:
        """Convert sentiment score to signal.

        Args:
            score: Sentiment score (0-1)

        Returns:
            Signal string: "bearish", "neutral", or "bullish"
        """
        if score is None:
            return None

        if score < 0.3:
            return "bearish"
        elif score > 0.7:
            return "bullish"
        else:
            return "neutral"

    def _is_cache_valid(self) -> bool:
        """Check if cached sentiment is still valid.

        Returns:
            True if cache is valid
        """
        if self._cached_sentiment is None or self._cache_timestamp is None:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < self.cache_ttl

    def clear_cache(self):
        """Clear cached sentiment."""
        self._cached_sentiment = None
        self._cache_timestamp = None
        self.logger.debug("Sentiment cache cleared")

    def get_active_sources(self) -> list:
        """Get list of active sentiment sources.

        Returns:
            List of enabled source names
        """
        sources = []
        if self.enable_twitter:
            sources.append('twitter')
        if self.enable_news:
            sources.append('news')
        if self.enable_onchain:
            sources.append('onchain')
        if self.enable_fear_greed:
            sources.append('fear_greed')

        return sources

    def update_weights(self, **weights):
        """Update sentiment source weights.

        Args:
            **weights: New weights for sources
        """
        # Update weights
        for source, weight in weights.items():
            if source in self.weights:
                self.weights[source] = weight

        # Renormalize
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Clear cache (weights changed)
        self.clear_cache()

        self.logger.info(f"Weights updated: {self.weights}")

    def __repr__(self) -> str:
        """String representation."""
        sources = ', '.join(self.get_active_sources())
        return f"SentimentAggregator(symbol={self.symbol}, sources=[{sources}])"
