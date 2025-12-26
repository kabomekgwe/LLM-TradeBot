"""Twitter Sentiment Analysis - Social media sentiment tracking.

Analyzes Twitter/X sentiment for cryptocurrency symbols using:
- Tweet volume and engagement metrics
- Natural language sentiment analysis (VADER)
- Trending hashtags and mentions
- Influential account activity

Requires Twitter API v2 Bearer Token.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class TwitterSentiment:
    """Twitter/X sentiment analyzer for cryptocurrency.

    Collects and analyzes tweets about a specific cryptocurrency
    to determine market sentiment.

    Setup:
        1. Apply for Twitter Developer Account
        2. Create app at https://developer.twitter.com
        3. Generate Bearer Token
        4. Set TWITTER_BEARER_TOKEN environment variable

    Example:
        >>> twitter = TwitterSentiment("BTC")
        >>> score = await twitter.get_sentiment_score()
        >>> print(f"Sentiment: {score:.2f}")
    """

    def __init__(
        self,
        symbol: str,
        bearer_token: Optional[str] = None,
        max_tweets: int = 100,
        time_window_hours: int = 24,
    ):
        """Initialize Twitter sentiment analyzer.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            bearer_token: Twitter API Bearer Token
            max_tweets: Maximum tweets to analyze
            time_window_hours: Time window for tweet collection
        """
        if not TWEEPY_AVAILABLE:
            raise ImportError("tweepy is required. Install with: pip install tweepy")

        if not VADER_AVAILABLE:
            raise ImportError("vaderSentiment is required. Install with: pip install vaderSentiment")

        self.logger = logging.getLogger(__name__)

        self.symbol = symbol.upper()
        self.max_tweets = max_tweets
        self.time_window = timedelta(hours=time_window_hours)

        # Twitter API client
        if bearer_token is None:
            import os
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        if not bearer_token:
            raise ValueError("Twitter Bearer Token not provided. Set TWITTER_BEARER_TOKEN environment variable.")

        try:
            self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        except Exception as e:
            raise ValueError(f"Failed to initialize Twitter client: {e}")

        # VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # Search queries
        self.queries = self._build_search_queries()

    def _build_search_queries(self) -> List[str]:
        """Build Twitter search queries for the symbol.

        Returns:
            List of search query strings
        """
        # Common cryptocurrency terms
        symbol_queries = [
            f"${self.symbol}",  # Cashtag
            f"#{self.symbol}",  # Hashtag
        ]

        # Add full names for major cryptocurrencies
        full_names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'ADA': 'Cardano',
            'DOGE': 'Dogecoin',
            'XRP': 'Ripple',
            'DOT': 'Polkadot',
            'MATIC': 'Polygon',
            'AVAX': 'Avalanche',
            'LINK': 'Chainlink',
        }

        if self.symbol in full_names:
            symbol_queries.append(full_names[self.symbol])

        return symbol_queries

    async def get_sentiment_score(self) -> float:
        """Get aggregated Twitter sentiment score.

        Returns:
            Sentiment score (0-1, where 0.5 is neutral)
        """
        self.logger.info(f"Fetching Twitter sentiment for {self.symbol}...")

        # Collect tweets
        tweets = await self._fetch_tweets()

        if not tweets:
            self.logger.warning("No tweets found, returning neutral sentiment")
            return 0.5

        # Analyze sentiment
        sentiments = [self._analyze_tweet_sentiment(tweet) for tweet in tweets]

        # Calculate aggregate score
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Convert from [-1, 1] to [0, 1]
        normalized_score = (avg_sentiment + 1) / 2

        self.logger.info(
            f"Twitter sentiment for {self.symbol}: {normalized_score:.2f} "
            f"({len(tweets)} tweets analyzed)"
        )

        return normalized_score

    async def _fetch_tweets(self) -> List[Dict[str, Any]]:
        """Fetch recent tweets about the symbol.

        Returns:
            List of tweet dictionaries
        """
        all_tweets = []

        # Calculate time window
        start_time = datetime.utcnow() - self.time_window

        for query in self.queries:
            try:
                # Search tweets
                response = self.client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",
                    max_results=min(self.max_tweets, 100),
                    start_time=start_time,
                    tweet_fields=['created_at', 'public_metrics', 'lang'],
                )

                if response.data:
                    for tweet in response.data:
                        all_tweets.append({
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
                        })

            except tweepy.TweepyException as e:
                self.logger.error(f"Twitter API error for query '{query}': {e}")
                continue

            except Exception as e:
                self.logger.error(f"Failed to fetch tweets for query '{query}': {e}")
                continue

        # Remove duplicates
        unique_tweets = {tweet['id']: tweet for tweet in all_tweets}.values()

        self.logger.debug(f"Fetched {len(unique_tweets)} unique tweets")

        return list(unique_tweets)

    def _analyze_tweet_sentiment(self, tweet: Dict[str, Any]) -> float:
        """Analyze sentiment of a single tweet.

        Args:
            tweet: Tweet dictionary

        Returns:
            Sentiment score (-1 to 1)
        """
        text = tweet['text']

        # Clean tweet text
        text = self._clean_tweet_text(text)

        # VADER sentiment analysis
        scores = self.vader.polarity_scores(text)

        # Compound score is the aggregate (-1 to 1)
        sentiment = scores['compound']

        # Apply engagement weighting (more engagement = more weight)
        if 'metrics' in tweet and tweet['metrics']:
            metrics = tweet['metrics']
            likes = metrics.get('like_count', 0)
            retweets = metrics.get('retweet_count', 0)
            replies = metrics.get('reply_count', 0)

            # Calculate engagement score
            engagement = likes + (retweets * 2) + replies

            # Weight sentiment by engagement (capped at 5x)
            weight = min(1 + (engagement / 100), 5)
            sentiment *= weight

        return sentiment

    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis.

        Args:
            text: Raw tweet text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    async def get_detailed_sentiment(self) -> Dict[str, Any]:
        """Get detailed sentiment analysis with breakdown.

        Returns:
            Dictionary with detailed sentiment metrics
        """
        tweets = await self._fetch_tweets()

        if not tweets:
            return {
                'symbol': self.symbol,
                'score': 0.5,
                'signal': 'neutral',
                'tweet_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
            }

        # Analyze all tweets
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for tweet in tweets:
            sentiment = self._analyze_tweet_sentiment(tweet)
            sentiments.append(sentiment)

            if sentiment > 0.05:
                positive_count += 1
            elif sentiment < -0.05:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate aggregate
        avg_sentiment = sum(sentiments) / len(sentiments)
        normalized_score = (avg_sentiment + 1) / 2

        return {
            'symbol': self.symbol,
            'score': normalized_score,
            'signal': 'bullish' if normalized_score > 0.6 else ('bearish' if normalized_score < 0.4 else 'neutral'),
            'tweet_count': len(tweets),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_pct': (positive_count / len(tweets)) * 100,
            'negative_pct': (negative_count / len(tweets)) * 100,
            'neutral_pct': (neutral_count / len(tweets)) * 100,
            'timestamp': datetime.now().isoformat(),
        }

    def get_trending_topics(self) -> List[str]:
        """Get trending topics related to the symbol.

        Returns:
            List of trending hashtags/topics
        """
        # Note: This requires Twitter API v2 with elevated access
        # Placeholder implementation
        self.logger.warning("Trending topics requires elevated Twitter API access")
        return []

    def __repr__(self) -> str:
        """String representation."""
        return f"TwitterSentiment(symbol={self.symbol}, max_tweets={self.max_tweets})"
