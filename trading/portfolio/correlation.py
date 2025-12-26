"""Correlation Analyzer - Asset correlation and dependency tracking.

Analyzes correlations between portfolio assets to:
- Identify diversification opportunities
- Detect concentration risk
- Optimize asset selection
- Monitor correlation changes over time
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class CorrelationAnalyzer:
    """Analyze correlations between portfolio assets.

    Tracks price correlations to assess diversification
    and identify concentration risks.

    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> await analyzer.update_prices({
        ...     "BTC/USDT": [42000, 42500, 43000],
        ...     "ETH/USDT": [2200, 2250, 2300],
        ... })
        >>> corr_matrix = analyzer.get_correlation_matrix()
    """

    def __init__(
        self,
        window_days: int = 30,
        min_samples: int = 20,
    ):
        """Initialize correlation analyzer.

        Args:
            window_days: Rolling window for correlation calculation
            min_samples: Minimum samples required for correlation
        """
        self.logger = logging.getLogger(__name__)

        self.window_days = window_days
        self.min_samples = min_samples

        # Price history: symbol -> list of (timestamp, price)
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def add_price_update(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """Add price update for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append((timestamp, price))

        # Keep only recent data (window + buffer)
        cutoff = datetime.now() - timedelta(days=self.window_days * 2)
        self.price_history[symbol] = [
            (ts, p) for ts, p in self.price_history[symbol]
            if ts > cutoff
        ]

    def batch_update_prices(self, prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """Add price updates for multiple symbols.

        Args:
            prices: Dictionary mapping symbols to prices
            timestamp: Timestamp (default: now)
        """
        for symbol, price in prices.items():
            self.add_price_update(symbol, price, timestamp)

    def get_correlation_matrix(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get correlation matrix for portfolio assets.

        Args:
            symbols: List of symbols to include (None = all)

        Returns:
            DataFrame with correlation matrix
        """
        if symbols is None:
            symbols = list(self.price_history.keys())

        # Build price series for each symbol
        price_series = {}

        for symbol in symbols:
            if symbol not in self.price_history:
                continue

            history = self.price_history[symbol]

            if len(history) < self.min_samples:
                self.logger.warning(f"Insufficient data for {symbol}: {len(history)} samples")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(history, columns=['timestamp', 'price'])
            df = df.set_index('timestamp')
            df = df.sort_index()

            # Resample to daily (if multiple entries per day)
            df = df.resample('D').last().dropna()

            price_series[symbol] = df['price']

        if len(price_series) < 2:
            self.logger.warning("Need at least 2 symbols for correlation")
            return pd.DataFrame()

        # Combine into single DataFrame
        df = pd.DataFrame(price_series)

        # Calculate returns
        returns = df.pct_change().dropna()

        # Correlation matrix
        corr_matrix = returns.corr()

        return corr_matrix

    def get_correlation_pair(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient (-1 to 1)
        """
        corr_matrix = self.get_correlation_matrix([symbol1, symbol2])

        if corr_matrix.empty or symbol1 not in corr_matrix.columns or symbol2 not in corr_matrix.columns:
            return 0.0

        return corr_matrix.loc[symbol1, symbol2]

    def get_average_correlation(self, symbol: str, other_symbols: Optional[List[str]] = None) -> float:
        """Get average correlation of a symbol with others.

        Args:
            symbol: Symbol to analyze
            other_symbols: Symbols to compare with (None = all others)

        Returns:
            Average correlation
        """
        if other_symbols is None:
            other_symbols = [s for s in self.price_history.keys() if s != symbol]

        if not other_symbols:
            return 0.0

        corr_matrix = self.get_correlation_matrix([symbol] + other_symbols)

        if corr_matrix.empty or symbol not in corr_matrix.columns:
            return 0.0

        # Get correlations with other symbols
        correlations = [corr_matrix.loc[symbol, other] for other in other_symbols if other in corr_matrix.columns]

        if not correlations:
            return 0.0

        return np.mean(correlations)

    def find_diversification_candidates(
        self,
        existing_symbols: List[str],
        candidate_symbols: List[str],
        max_correlation: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Find symbols that provide diversification.

        Args:
            existing_symbols: Current portfolio symbols
            candidate_symbols: Potential new symbols to add
            max_correlation: Maximum acceptable correlation

        Returns:
            List of (symbol, avg_correlation) tuples, sorted by diversification
        """
        candidates = []

        for candidate in candidate_symbols:
            if candidate in existing_symbols:
                continue

            avg_corr = self.get_average_correlation(candidate, existing_symbols)

            if avg_corr <= max_correlation:
                candidates.append((candidate, avg_corr))

        # Sort by lowest correlation (best diversification)
        candidates.sort(key=lambda x: x[1])

        return candidates

    def calculate_portfolio_correlation(self, allocation: Dict[str, float]) -> float:
        """Calculate weighted average correlation for a portfolio.

        Args:
            allocation: Portfolio allocation (symbol -> weight)

        Returns:
            Weighted average correlation
        """
        symbols = list(allocation.keys())
        corr_matrix = self.get_correlation_matrix(symbols)

        if corr_matrix.empty:
            return 0.0

        # Calculate weighted correlation
        total_corr = 0.0
        total_weight = 0.0

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j or symbol1 not in corr_matrix.columns or symbol2 not in corr_matrix.columns:
                    continue

                weight = allocation[symbol1] * allocation[symbol2]
                corr = corr_matrix.loc[symbol1, symbol2]

                total_corr += weight * corr
                total_weight += weight

        if total_weight == 0:
            return 0.0

        avg_corr = total_corr / total_weight

        return avg_corr

    def detect_correlation_clusters(
        self,
        symbols: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[List[str]]:
        """Detect groups of highly correlated symbols.

        Args:
            symbols: Symbols to analyze (None = all)
            threshold: Correlation threshold for clustering

        Returns:
            List of symbol clusters
        """
        corr_matrix = self.get_correlation_matrix(symbols)

        if corr_matrix.empty:
            return []

        # Simple clustering based on correlation threshold
        clusters = []
        assigned = set()

        for symbol1 in corr_matrix.columns:
            if symbol1 in assigned:
                continue

            # Start new cluster
            cluster = [symbol1]
            assigned.add(symbol1)

            # Add highly correlated symbols
            for symbol2 in corr_matrix.columns:
                if symbol2 == symbol1 or symbol2 in assigned:
                    continue

                corr = corr_matrix.loc[symbol1, symbol2]
                if corr >= threshold:
                    cluster.append(symbol2)
                    assigned.add(symbol2)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def get_correlation_report(self) -> Dict[str, any]:
        """Get comprehensive correlation analysis report.

        Returns:
            Dictionary with correlation metrics
        """
        symbols = list(self.price_history.keys())
        corr_matrix = self.get_correlation_matrix(symbols)

        if corr_matrix.empty:
            return {}

        # Calculate statistics
        correlations = []
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:
                    continue
                if symbol1 in corr_matrix.columns and symbol2 in corr_matrix.columns:
                    correlations.append(corr_matrix.loc[symbol1, symbol2])

        return {
            'num_symbols': len(symbols),
            'avg_correlation': np.mean(correlations) if correlations else 0.0,
            'max_correlation': np.max(correlations) if correlations else 0.0,
            'min_correlation': np.min(correlations) if correlations else 0.0,
            'correlation_matrix': corr_matrix.to_dict(),
            'clusters': self.detect_correlation_clusters(symbols),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"CorrelationAnalyzer(symbols={len(self.price_history)}, window={self.window_days}d)"
