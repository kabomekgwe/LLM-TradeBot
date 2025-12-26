"""Market Sentiment Analysis - Multi-source sentiment aggregation.

Collects and analyzes market sentiment from multiple sources:
- Twitter/X: Social media sentiment and trends
- News: Financial news sentiment with FinBERT
- On-chain: Blockchain metrics from Glassnode
- Fear & Greed Index: Market sentiment indicator

Provides aggregated sentiment score for trading decisions.

Components:
- SentimentAggregator: Combine sentiment from all sources
- TwitterSentiment: X/Twitter sentiment analysis
- NewsSentiment: Financial news sentiment
- OnChainMetrics: Blockchain activity metrics
- FearGreedIndex: Market fear & greed indicator

Example Usage:
    ```python
    from trading.sentiment import SentimentAggregator

    # Initialize aggregator
    aggregator = SentimentAggregator(
        symbol="BTC",
        enable_twitter=True,
        enable_news=True,
        enable_onchain=True,
        enable_fear_greed=True
    )

    # Get aggregated sentiment
    sentiment = await aggregator.get_sentiment()

    print(f"Overall Sentiment: {sentiment['overall_score']:.2f}")
    print(f"Twitter: {sentiment['twitter_score']:.2f}")
    print(f"News: {sentiment['news_score']:.2f}")
    print(f"On-chain: {sentiment['onchain_score']:.2f}")
    print(f"Fear & Greed: {sentiment['fear_greed_score']:.2f}")

    # Get sentiment signal (-1 to 1)
    signal = sentiment['overall_score']
    if signal > 0.6:
        print("Strong bullish sentiment")
    elif signal < 0.4:
        print("Strong bearish sentiment")
    else:
        print("Neutral sentiment")
    ```

Configuration:
    Set in TradingConfig or environment:
    ```bash
    # Enable sentiment analysis
    SENTIMENT_ENABLED=true

    # Twitter/X API
    TWITTER_BEARER_TOKEN=your_bearer_token

    # News API
    NEWS_API_KEY=your_news_api_key

    # Glassnode (on-chain)
    GLASSNODE_API_KEY=your_glassnode_key

    # Sentiment weights (how much to trust each source)
    SENTIMENT_TWITTER_WEIGHT=0.25
    SENTIMENT_NEWS_WEIGHT=0.35
    SENTIMENT_ONCHAIN_WEIGHT=0.20
    SENTIMENT_FEAR_GREED_WEIGHT=0.20
    ```

Sentiment Scores:
    All scores normalized to 0-1 range:
    - 0.0 - 0.3: Bearish
    - 0.3 - 0.7: Neutral
    - 0.7 - 1.0: Bullish

API Setup:
    **Twitter/X:**
    1. Apply for Twitter Developer Account
    2. Create app and get Bearer Token
    3. Set TWITTER_BEARER_TOKEN

    **News API:**
    1. Sign up at https://newsapi.org
    2. Get free API key (100 requests/day)
    3. Set NEWS_API_KEY

    **Glassnode:**
    1. Sign up at https://glassnode.com
    2. Get API key from settings
    3. Set GLASSNODE_API_KEY
"""

from .aggregator import SentimentAggregator
from .twitter import TwitterSentiment
from .news import NewsSentiment
from .onchain import OnChainMetrics
from .fear_greed import FearGreedIndex

__all__ = [
    "SentimentAggregator",
    "TwitterSentiment",
    "NewsSentiment",
    "OnChainMetrics",
    "FearGreedIndex",
]

__version__ = "1.0.0"
