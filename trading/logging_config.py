"""
Structured logging configuration for LLM-TradeBot.

Provides JSON-formatted logs with contextual information for production observability.
"""

import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_logging(level: str = "INFO", use_json: bool = True) -> logging.Logger:
    """
    Configure structured logging for the trading bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: If True, output JSON format; if False, use standard text format

    Returns:
        Configured root logger

    Example:
        >>> logger = setup_logging(level="INFO", use_json=True)
        >>> logger.info("trade_executed", extra={"symbol": "BTC/USDT", "side": "buy"})
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler (stdout for container compatibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Configure formatter
    if use_json:
        # JSON formatter for production
        json_formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )
        console_handler.setFormatter(json_formatter)
    else:
        # Standard formatter for development
        text_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(text_formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("agent_vote", extra={"agent": "bull", "vote": 1.0})
    """
    return logging.getLogger(name)


# Context utilities for decision tracing
class DecisionContext:
    """
    Thread-safe context for decision tracing.

    Allows adding decision_id to all logs within a trading loop.
    """

    _context = {}

    @classmethod
    def set_decision_id(cls, decision_id: str):
        """Set decision ID for current context."""
        cls._context["decision_id"] = decision_id

    @classmethod
    def get_decision_id(cls) -> str | None:
        """Get current decision ID."""
        return cls._context.get("decision_id")

    @classmethod
    def clear(cls):
        """Clear context."""
        cls._context.clear()

    @classmethod
    def get_extra(cls) -> dict:
        """Get extra dict with decision_id for logging."""
        if decision_id := cls.get_decision_id():
            return {"decision_id": decision_id}
        return {}
