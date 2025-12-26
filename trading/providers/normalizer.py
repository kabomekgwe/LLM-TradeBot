"""Exchange-specific normalization utilities.

Handles differences in symbol formats, timeframes, and other
exchange-specific quirks across different platforms.
"""

from typing import Optional


class ExchangeNormalizer:
    """Normalizes exchange-specific differences.

    Different exchanges use different formats for the same concepts:
    - Symbols: BTC/USDT vs BTCUSDT vs BTC-USD
    - Timeframes: 5m vs 5min vs 300
    - Order sides: buy/sell vs long/short
    """

    # Symbol format templates for each exchange
    SYMBOL_FORMATS = {
        "binance_futures": "{base}/{quote}",
        "binance_spot": "{base}/{quote}",
        "kraken": "{base}/{quote}",
        "coinbase": "{base}-{quote}",
        "alpaca": "{base}",  # Stocks use ticker only
        "paper": "{base}/{quote}",  # Same as Binance
    }

    # Timeframe mapping (standard → exchange-specific)
    TIMEFRAME_MAPPINGS = {
        "binance_futures": {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        },
        "alpaca": {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "1h": "1Hour",
            "4h": "4Hour",
            "1d": "1Day",
        },
    }

    @staticmethod
    def normalize_symbol(
        symbol: str,
        from_exchange: str,
        to_exchange: str
    ) -> str:
        """Convert symbol between exchange formats.

        Args:
            symbol: Symbol in source format (e.g., "BTC/USDT")
            from_exchange: Source exchange name
            to_exchange: Target exchange name

        Returns:
            Symbol in target format

        Example:
            >>> ExchangeNormalizer.normalize_symbol(
            ...     "BTC/USDT", "binance_futures", "coinbase"
            ... )
            'BTC-USD'
        """
        # Parse symbol to extract base and quote
        base, quote = ExchangeNormalizer._parse_symbol(symbol, from_exchange)

        # Format for target exchange
        template = ExchangeNormalizer.SYMBOL_FORMATS.get(
            to_exchange,
            "{base}/{quote}"  # Default format
        )

        return template.format(base=base, quote=quote)

    @staticmethod
    def _parse_symbol(symbol: str, exchange: str) -> tuple[str, str]:
        """Parse symbol into base and quote currencies.

        Args:
            symbol: Symbol to parse
            exchange: Exchange name for context

        Returns:
            (base, quote) tuple

        Example:
            >>> ExchangeNormalizer._parse_symbol("BTC/USDT", "binance_futures")
            ('BTC', 'USDT')
        """
        # Handle different formats
        if "/" in symbol:
            base, quote = symbol.split("/")
        elif "-" in symbol:
            base, quote = symbol.split("-")
        else:
            # No separator (like Alpaca stocks)
            base = symbol
            quote = ""

        return base.upper(), quote.upper()

    @staticmethod
    def normalize_timeframe(
        timeframe: str,
        exchange: str
    ) -> str:
        """Convert timeframe to exchange-specific format.

        Args:
            timeframe: Standard timeframe (e.g., "5m", "1h")
            exchange: Exchange name

        Returns:
            Exchange-specific timeframe

        Example:
            >>> ExchangeNormalizer.normalize_timeframe("5m", "alpaca")
            '5Min'
        """
        mappings = ExchangeNormalizer.TIMEFRAME_MAPPINGS.get(exchange, {})
        return mappings.get(timeframe, timeframe)  # Return as-is if no mapping

    @staticmethod
    def standardize_symbol(symbol: str, exchange: str) -> str:
        """Convert exchange-specific symbol to standard format.

        Standard format is BASE/QUOTE (e.g., "BTC/USDT").

        Args:
            symbol: Exchange-specific symbol
            exchange: Exchange name

        Returns:
            Standardized symbol

        Example:
            >>> ExchangeNormalizer.standardize_symbol("BTC-USD", "coinbase")
            'BTC/USD'
        """
        base, quote = ExchangeNormalizer._parse_symbol(symbol, exchange)

        if quote:
            return f"{base}/{quote}"
        else:
            return base

    @staticmethod
    def get_quote_currency(symbol: str, exchange: str) -> str:
        """Extract quote currency from symbol.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            Quote currency (e.g., "USDT", "USD")

        Example:
            >>> ExchangeNormalizer.get_quote_currency("BTC/USDT", "binance_futures")
            'USDT'
        """
        _, quote = ExchangeNormalizer._parse_symbol(symbol, exchange)
        return quote

    @staticmethod
    def get_base_currency(symbol: str, exchange: str) -> str:
        """Extract base currency from symbol.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            Base currency (e.g., "BTC")

        Example:
            >>> ExchangeNormalizer.get_base_currency("BTC/USDT", "binance_futures")
            'BTC'
        """
        base, _ = ExchangeNormalizer._parse_symbol(symbol, exchange)
        return base


class PriceNormalizer:
    """Normalizes prices and amounts across exchanges.

    Some exchanges use different precision or scaling.
    """

    @staticmethod
    def round_price(price: float, symbol: str, exchange: str) -> float:
        """Round price to exchange-specific precision.

        Args:
            price: Price to round
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            Rounded price

        Note:
            This is a simplified implementation. In production, would
            query exchange markets info for actual precision requirements.
        """
        # Default precision by exchange
        precision_map = {
            "binance_futures": 2,
            "binance_spot": 2,
            "kraken": 1,
            "coinbase": 2,
            "alpaca": 2,
        }

        precision = precision_map.get(exchange, 2)
        return round(price, precision)

    @staticmethod
    def round_amount(amount: float, symbol: str, exchange: str) -> float:
        """Round amount/size to exchange-specific precision.

        Args:
            amount: Amount to round
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            Rounded amount
        """
        # Default amount precision by exchange
        precision_map = {
            "binance_futures": 4,
            "binance_spot": 4,
            "kraken": 5,
            "coinbase": 8,
            "alpaca": 0,  # Stocks are whole shares
        }

        precision = precision_map.get(exchange, 4)
        return round(amount, precision)
