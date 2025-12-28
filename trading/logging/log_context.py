"""Logging context management with correlation IDs.

This module provides context-aware logging with correlation IDs for tracing
requests across the system. Uses contextvars for thread-safe context propagation.

Features:
- Thread-safe correlation ID storage using contextvars
- Automatic correlation ID generation
- Logging filter to inject correlation IDs into log records
- Manual correlation ID setting for custom workflows

Usage:
    from trading.logging.log_context import LogContext

    # Generate and set correlation ID
    correlation_id = LogContext.generate_correlation_id()
    LogContext.set_correlation_id(correlation_id)

    # All logs in this context will include the correlation ID
    logger.info("Processing trade")

    # Retrieve current correlation ID
    current_id = LogContext.get_correlation_id()
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional


# Context variable for correlation ID
# This is thread-safe and propagates through async contexts
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class LogContext:
    """Manage logging context with correlation IDs."""

    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set correlation ID for current context.

        Args:
            correlation_id: Correlation ID to set

        Example:
            >>> LogContext.set_correlation_id("req-12345")
            >>> logger.info("Processing request")  # Will include correlation_id
        """
        correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get correlation ID from current context.

        Returns:
            Correlation ID or None if not set

        Example:
            >>> LogContext.set_correlation_id("req-12345")
            >>> current_id = LogContext.get_correlation_id()
            >>> print(current_id)
            req-12345
        """
        return correlation_id_var.get()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate new correlation ID.

        Returns:
            UUID4 string as correlation ID

        Example:
            >>> correlation_id = LogContext.generate_correlation_id()
            >>> LogContext.set_correlation_id(correlation_id)
        """
        return str(uuid.uuid4())


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records.

    This filter automatically injects the current correlation ID
    from context into every log record as the 'correlation_id' attribute.

    Example:
        >>> filter = CorrelationFilter()
        >>> handler.addFilter(filter)
        >>> LogContext.set_correlation_id("req-12345")
        >>> logger.info("Test")  # Log will include correlation_id="req-12345"
    """

    def filter(self, record):
        """Add correlation ID to log record.

        Args:
            record: LogRecord to modify

        Returns:
            True (always process the log record)
        """
        correlation_id = LogContext.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True
