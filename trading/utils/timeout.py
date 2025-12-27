"""
Timeout utilities for async operations.

Provides decorators and context managers for adding timeouts to async functions.
"""

import asyncio
import functools
from typing import TypeVar, Callable, Any

from trading.exceptions import AgentTimeoutError
from trading.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def with_timeout(seconds: float, error_class=AgentTimeoutError):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds
        error_class: Exception class to raise on timeout (default: AgentTimeoutError)

    Returns:
        Decorated async function

    Example:
        @with_timeout(30.0)
        async def fetch_data(symbol: str):
            return await exchange.fetch_ohlcv(symbol)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                func_name = func.__name__
                logger.error(
                    "function_timeout",
                    extra={
                        "function": func_name,
                        "timeout_seconds": seconds,
                        "args": str(args)[:100],  # Truncate for logging
                    },
                )
                raise error_class(f"{func_name} exceeded {seconds}s timeout")

        return wrapper

    return decorator


async def with_timeout_and_retry(
    coro, timeout: float, retries: int = 1, retry_delay: float = 1.0
) -> Any:
    """
    Execute async function with timeout and retry logic.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        retries: Number of retry attempts (default: 1)
        retry_delay: Delay between retries in seconds (default: 1.0)

    Returns:
        Result from coroutine

    Raises:
        AgentTimeoutError: If all retries exhausted

    Example:
        result = await with_timeout_and_retry(
            exchange.fetch_ohlcv(symbol, "5m"),
            timeout=30.0,
            retries=2
        )
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as e:
            last_error = e
            if attempt < retries:
                logger.warning(
                    "timeout_retry",
                    extra={
                        "attempt": attempt + 1,
                        "retries": retries,
                        "timeout_seconds": timeout,
                        "retry_delay": retry_delay,
                    },
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "timeout_exhausted",
                    extra={
                        "attempts": attempt + 1,
                        "timeout_seconds": timeout,
                    },
                )

    raise AgentTimeoutError(f"All {retries + 1} attempts exhausted (timeout: {timeout}s)")
