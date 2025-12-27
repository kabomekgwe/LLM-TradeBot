"""Market microstructure features from CCXT order book data.

Extracts order book depth, bid-ask spread, trade imbalance, and mid-price
from exchange L2 order book data via CCXT.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Extract market microstructure features from CCXT order book."""

    def __init__(self, depth_levels: int = 10):
        """Initialize microstructure feature extractor.

        Args:
            depth_levels: Number of order book levels to analyze (default 10)
        """
        self.depth_levels = depth_levels

    async def extract_features(self, exchange, symbol: str) -> Dict[str, float]:
        """Extract microstructure features from current order book.

        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dictionary with keys: bid_ask_spread, spread_pct, mid_price,
                                 order_book_imbalance, bid_volume, ask_volume

        Raises:
            Exception: If order book fetch fails
        """
        try:
            # Fetch order book with depth
            order_book = await exchange.fetch_order_book(symbol, limit=self.depth_levels * 2)

            if not order_book['bids'] or not order_book['asks']:
                raise ValueError(f"Empty order book for {symbol}")

            # Best bid/ask prices
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]

            # Bid-ask spread
            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100

            # Mid-price
            mid_price = (best_bid + best_ask) / 2

            # Order book imbalance (top N levels)
            bid_volume = sum([level[1] for level in order_book['bids'][:self.depth_levels]])
            ask_volume = sum([level[1] for level in order_book['asks'][:self.depth_levels]])

            # Normalized imbalance: -1 (all asks) to +1 (all bids)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0

            logger.debug(f"Microstructure features for {symbol}: spread={spread_pct:.4f}%, imbalance={imbalance:.4f}")

            return {
                'bid_ask_spread': spread,
                'spread_pct': spread_pct,
                'mid_price': mid_price,
                'order_book_imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
            }

        except Exception as e:
            logger.error(f"Failed to extract microstructure features for {symbol}: {e}")
            raise
