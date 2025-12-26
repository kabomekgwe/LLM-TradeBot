"""News Sentiment Analysis - Financial news sentiment tracking.

Analyzes cryptocurrency news sentiment using:
- News API for article collection
- VADER sentiment analysis for general sentiment
- FinBERT for financial-specific sentiment (optional)
- Source credibility weighting

Requires News API key (free tier: 100 requests/day).
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class NewsSentiment:
    """News sentiment analyzer for cryptocurrency.

    Collects and analyzes financial news articles about a cryptocurrency
    to determine market sentiment.

    Setup:
        1. Sign up at https://newsapi.org
        2. Get free API key (100 requests/day)
        3. Set NEWS_API_KEY environment variable

    Example:
        >>> news = NewsSentiment("BTC")
        >>> score = await news.get_sentiment_score()
        >>> print(f"News sentiment: {score:.2f}")
    """

    def __init__(
        self,
        symbol: str,
        api_key: Optional[str] = None,
        max_articles: int = 50,
        time_window_hours: int = 48,
        use_finbert: bool = False,
    ):
        """Initialize news sentiment analyzer.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            api_key: News API key
            max_articles: Maximum articles to analyze
            time_window_hours: Time window for article collection
            use_finbert: Use FinBERT for financial sentiment (requires transformers)
        """
        if not NEWSAPI_AVAILABLE:
            raise ImportError("newsapi-python is required. Install with: pip install newsapi-python")

        if not VADER_AVAILABLE:
            raise ImportError("vaderSentiment is required. Install with: pip install vaderSentiment")

        self.logger = logging.getLogger(__name__)

        self.symbol = symbol.upper()
        self.max_articles = max_articles
        self.time_window = timedelta(hours=time_window_hours)
        self.use_finbert = use_finbert

        # News API client
        if api_key is None:
            import os
            api_key = os.getenv("NEWS_API_KEY")

        if not api_key:
            raise ValueError("News API key not provided. Set NEWS_API_KEY environment variable.")

        try:
            self.client = NewsApiClient(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize News API client: {e}")

        # VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # FinBERT (optional)
        self.finbert = None
        if use_finbert:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.finbert = True
                self.logger.info("FinBERT loaded successfully")

            except ImportError:
                self.logger.warning("transformers not available. Using VADER only.")
                self.use_finbert = False

            except Exception as e:
                self.logger.error(f"Failed to load FinBERT: {e}. Using VADER only.")
                self.use_finbert = False

        # Search keywords
        self.keywords = self._build_search_keywords()

        # Source credibility weights
        self.source_weights = {
            'bloomberg.com': 1.5,
            'reuters.com': 1.5,
            'coindesk.com': 1.3,
            'cointelegraph.com': 1.2,
            'decrypt.co': 1.2,
            'cnbc.com': 1.3,
            'wsj.com': 1.4,
            'ft.com': 1.4,
        }

    def _build_search_keywords(self) -> List[str]:
        """Build search keywords for the symbol.

        Returns:
            List of search keywords
        """
        keywords = [self.symbol]

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
            keywords.append(full_names[self.symbol])

        return keywords

    async def get_sentiment_score(self) -> float:
        """Get aggregated news sentiment score.

        Returns:
            Sentiment score (0-1, where 0.5 is neutral)
        """
        self.logger.info(f"Fetching news sentiment for {self.symbol}...")

        # Collect articles
        articles = await self._fetch_articles()

        if not articles:
            self.logger.warning("No articles found, returning neutral sentiment")
            return 0.5

        # Analyze sentiment
        sentiments = []
        for article in articles:
            sentiment = self._analyze_article_sentiment(article)
            sentiments.append(sentiment)

        # Calculate weighted average
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Convert from [-1, 1] to [0, 1]
        normalized_score = (avg_sentiment + 1) / 2

        self.logger.info(
            f"News sentiment for {self.symbol}: {normalized_score:.2f} "
            f"({len(articles)} articles analyzed)"
        )

        return normalized_score

    async def _fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch recent news articles about the symbol.

        Returns:
            List of article dictionaries
        """
        all_articles = []

        # Calculate time window
        from_date = (datetime.utcnow() - self.time_window).strftime('%Y-%m-%d')

        for keyword in self.keywords:
            try:
                # Search articles
                response = self.client.get_everything(
                    q=keyword,
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=min(self.max_articles, 100),
                )

                if response['status'] == 'ok' and response['articles']:
                    for article in response['articles']:
                        all_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                        })

            except Exception as e:
                self.logger.error(f"Failed to fetch news for keyword '{keyword}': {e}")
                continue

        # Remove duplicates based on URL
        unique_articles = {article['url']: article for article in all_articles if article['url']}.values()

        self.logger.debug(f"Fetched {len(unique_articles)} unique articles")

        return list(unique_articles)

    def _analyze_article_sentiment(self, article: Dict[str, Any]) -> float:
        """Analyze sentiment of a single article.

        Args:
            article: Article dictionary

        Returns:
            Sentiment score (-1 to 1)
        """
        # Combine title and description for analysis
        text = f"{article['title']} {article['description']}"

        if not text.strip():
            return 0.0

        # Use FinBERT if available, otherwise VADER
        if self.use_finbert and self.finbert:
            sentiment = self._analyze_with_finbert(text)
        else:
            sentiment = self._analyze_with_vader(text)

        # Apply source credibility weighting
        source_url = article.get('url', '')
        for domain, weight in self.source_weights.items():
            if domain in source_url:
                sentiment *= weight
                break

        return sentiment

    def _analyze_with_vader(self, text: str) -> float:
        """Analyze sentiment using VADER.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1 to 1)
        """
        scores = self.vader.polarity_scores(text)
        return scores['compound']

    def _analyze_with_finbert(self, text: str) -> float:
        """Analyze sentiment using FinBERT.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            import torch

            # Tokenize
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Predict
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT classes: [positive, negative, neutral]
            positive = probs[0][0].item()
            negative = probs[0][1].item()
            neutral = probs[0][2].item()

            # Calculate compound score
            sentiment = positive - negative

            return sentiment

        except Exception as e:
            self.logger.error(f"FinBERT analysis failed: {e}. Falling back to VADER.")
            return self._analyze_with_vader(text)

    async def get_detailed_sentiment(self) -> Dict[str, Any]:
        """Get detailed sentiment analysis with breakdown.

        Returns:
            Dictionary with detailed sentiment metrics
        """
        articles = await self._fetch_articles()

        if not articles:
            return {
                'symbol': self.symbol,
                'score': 0.5,
                'signal': 'neutral',
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
            }

        # Analyze all articles
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            sentiment = self._analyze_article_sentiment(article)
            sentiments.append(sentiment)

            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
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
            'article_count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_pct': (positive_count / len(articles)) * 100,
            'negative_pct': (negative_count / len(articles)) * 100,
            'neutral_pct': (neutral_count / len(articles)) * 100,
            'analyzer': 'FinBERT' if self.use_finbert and self.finbert else 'VADER',
            'timestamp': datetime.now().isoformat(),
        }

    def get_top_headlines(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get top recent headlines.

        Args:
            limit: Maximum number of headlines

        Returns:
            List of headline dictionaries
        """
        # Placeholder for top headlines
        # Would require separate API call
        return []

    def __repr__(self) -> str:
        """String representation."""
        analyzer = 'FinBERT' if self.use_finbert else 'VADER'
        return f"NewsSentiment(symbol={self.symbol}, analyzer={analyzer})"
