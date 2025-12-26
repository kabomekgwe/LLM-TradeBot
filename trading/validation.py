"""Input validation utilities for security boundaries.

This module provides validation functions to protect against malicious input
at all external boundaries (user commands, API parameters, market data).

Security Philosophy:
- Validate at every boundary (defense in depth)
- Fail fast with clear error messages
- Trust internal code, validate external input
- Prevent injection attacks and malformed data
"""

from typing import Any, Optional
import re


class ValidationError(Exception):
    """Raised when input validation fails.

    Use this instead of ValueError for validation errors to allow
    callers to distinguish validation failures from other errors.
    """
    pass


def validate_symbol(symbol: str) -> str:
    """Validate trading symbol format.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT", "ETH-USD")

    Returns:
        Validated symbol in uppercase

    Raises:
        ValidationError: If symbol format invalid

    Why: Prevents injection attacks via symbol parameter to exchange APIs.
         Malicious symbols could attempt SQL injection, command injection,
         or path traversal if not validated.

    Examples:
        >>> validate_symbol("BTC/USDT")
        "BTC/USDT"
        >>> validate_symbol("eth-usd")
        "ETH-USD"
        >>> validate_symbol("../../etc/passwd")  # Blocked
        ValidationError: Invalid symbol format
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Symbol must be a non-empty string")

    # Allow alphanumeric, slash, dash, underscore only
    # This blocks special chars that could be used for injection
    if not re.match(r'^[A-Z0-9/_-]+$', symbol.upper()):
        raise ValidationError(
            f"Invalid symbol format: {symbol}. "
            f"Only letters, numbers, /, -, and _ allowed."
        )

    if len(symbol) > 20:
        raise ValidationError(f"Symbol too long (max 20 chars): {symbol}")

    return symbol.upper()


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe string.

    Args:
        timeframe: Candle timeframe (e.g., "1m", "5m", "1h", "1d")

    Returns:
        Validated timeframe

    Raises:
        ValidationError: If timeframe invalid

    Why: Prevents unexpected values being passed to exchange APIs.
         Unknown timeframes could cause crashes or unexpected behavior.

    Examples:
        >>> validate_timeframe("1h")
        "1h"
        >>> validate_timeframe("999y")  # Blocked
        ValidationError: Invalid timeframe
    """
    valid_timeframes = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '12h',
        '1d', '1w', '1M'
    ]

    if timeframe not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe: {timeframe}. "
            f"Must be one of: {', '.join(valid_timeframes)}"
        )

    return timeframe


def validate_positive_number(value: Any, name: str = "value") -> float:
    """Validate positive number (for amounts, prices, limits).

    Args:
        value: Number to validate
        name: Parameter name for error messages

    Returns:
        Validated float value

    Raises:
        ValidationError: If not a positive number

    Why: Prevents negative amounts, zero prices, infinity values.
         These could cause math errors, invalid orders, or bypass risk limits.

    Examples:
        >>> validate_positive_number(100.5, "price")
        100.5
        >>> validate_positive_number(-10, "amount")  # Blocked
        ValidationError: amount must be positive
        >>> validate_positive_number(float('inf'), "price")  # Blocked
        ValidationError: price must be finite
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{name} must be a number, got: {type(value).__name__}"
        )

    if num <= 0:
        raise ValidationError(f"{name} must be positive, got: {num}")

    if not float('-inf') < num < float('inf'):
        raise ValidationError(f"{name} must be finite, got: {num}")

    return num


def validate_limit(limit: Optional[int], max_limit: int = 1000) -> Optional[int]:
    """Validate result limit parameter.

    Args:
        limit: Number of results to return
        max_limit: Maximum allowed limit

    Returns:
        Validated limit or None

    Raises:
        ValidationError: If limit invalid

    Why: Prevents memory exhaustion from excessive result sets.
         Attackers could request millions of records to DoS the application.

    Examples:
        >>> validate_limit(100)
        100
        >>> validate_limit(None)  # OK - means no limit
        None
        >>> validate_limit(999999)  # Blocked
        ValidationError: Limit cannot exceed 1000
    """
    if limit is None:
        return None

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValidationError(
            f"Limit must be an integer, got: {type(limit).__name__}"
        )

    if limit < 1:
        raise ValidationError(f"Limit must be at least 1, got: {limit}")

    if limit > max_limit:
        raise ValidationError(
            f"Limit cannot exceed {max_limit}, got: {limit}"
        )

    return limit


def validate_exchange_name(name: str) -> str:
    """Validate exchange name against supported providers.

    Args:
        name: Exchange provider name

    Returns:
        Validated exchange name

    Raises:
        ValidationError: If exchange not supported

    Why: Prevents arbitrary exchange names from being used.
         Could prevent bugs from typos or attempts to use unsupported exchanges.

    Examples:
        >>> validate_exchange_name("binance_futures")
        "binance_futures"
        >>> validate_exchange_name("random_exchange")  # Blocked
        ValidationError: Unknown exchange
    """
    valid_exchanges = [
        'binance_futures',
        'binance_spot',
        'kraken',
        'coinbase',
        'alpaca',
        'paper'
    ]

    if name not in valid_exchanges:
        raise ValidationError(
            f"Unknown exchange: {name}. "
            f"Supported: {', '.join(valid_exchanges)}"
        )

    return name


def validate_order_side(side: str) -> str:
    """Validate order side (buy/sell).

    Args:
        side: Order side

    Returns:
        Validated side in lowercase

    Raises:
        ValidationError: If side invalid

    Why: Prevents invalid order sides that could cause API errors.

    Examples:
        >>> validate_order_side("buy")
        "buy"
        >>> validate_order_side("BUY")
        "buy"
        >>> validate_order_side("invalid")  # Blocked
        ValidationError: Invalid order side
    """
    valid_sides = ['buy', 'sell', 'long', 'short']
    side_lower = side.lower()

    if side_lower not in valid_sides:
        raise ValidationError(
            f"Invalid order side: {side}. "
            f"Must be one of: {', '.join(valid_sides)}"
        )

    return side_lower


def validate_order_type(order_type: str) -> str:
    """Validate order type.

    Args:
        order_type: Order type

    Returns:
        Validated order type in lowercase

    Raises:
        ValidationError: If order type invalid

    Why: Prevents unsupported order types from being sent to exchange.

    Examples:
        >>> validate_order_type("market")
        "market"
        >>> validate_order_type("LIMIT")
        "limit"
        >>> validate_order_type("exotic")  # Blocked
        ValidationError: Invalid order type
    """
    valid_types = ['market', 'limit', 'stop_loss', 'take_profit']
    type_lower = order_type.lower()

    if type_lower not in valid_types:
        raise ValidationError(
            f"Invalid order type: {order_type}. "
            f"Must be one of: {', '.join(valid_types)}"
        )

    return type_lower
