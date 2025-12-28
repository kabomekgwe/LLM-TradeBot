"""Structured JSON logging configuration for production debugging.

This module configures structured JSON logging using python-json-logger,
enabling easy parsing and querying of logs in production environments.

Features:
- JSON formatted logs for stdout and file
- Automatic timestamp in ISO format
- Correlation IDs for request tracing
- Trade context (symbol, trade_id) in log records
- Configurable log levels
- Suppressed noisy third-party loggers

Usage:
    from trading.logging.json_logger import setup_json_logging

    # Initialize at application startup
    setup_json_logging(log_level="INFO")

    # Log with extra context
    logger.info("Trade executed", extra={
        'symbol': 'BTC/USDT',
        'trade_id': '12345',
        'price': 50000.0
    })
"""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log records.

        Args:
            log_record: Dictionary to be formatted as JSON
            record: LogRecord instance from logging module
            message_dict: Dictionary containing the log message
        """
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record['timestamp'] = self.formatTime(record, self.datefmt)

        # Add log level
        log_record['level'] = record.levelname

        # Add logger name
        log_record['logger'] = record.name

        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id

        # Add trade context if present
        if hasattr(record, 'symbol'):
            log_record['symbol'] = record.symbol
        if hasattr(record, 'trade_id'):
            log_record['trade_id'] = record.trade_id


def setup_json_logging(log_level: str = "INFO"):
    """Configure structured JSON logging for production.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> setup_json_logging(log_level="INFO")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started", extra={'version': '1.0.0'})
    """
    formatter = CustomJsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

    # Console handler (stdout for Docker logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler (optional, for persistent logs)
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / 'trading.json.log')
    file_handler.setFormatter(formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    logging.info("Structured JSON logging initialized", extra={'log_level': log_level})
