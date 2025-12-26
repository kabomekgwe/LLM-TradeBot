"""Base agent interface for trading agents.

All 8 agents in the adversarial decision system inherit from this base class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import TradingConfig
from ..providers.base import BaseExchangeProvider


class BaseAgent(ABC):
    """Base class for all trading agents.

    Provides common functionality and defines the interface that all
    agents must implement. Follows SRP - each agent does ONE thing.
    """

    def __init__(self, provider: BaseExchangeProvider, config: TradingConfig):
        """Initialize base agent.

        Args:
            provider: Exchange provider for market data access
            config: Trading configuration
        """
        self.provider = provider
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute agent logic.

        This is the main entry point for each agent. Takes a context dictionary
        containing all available data and returns an updated context with the
        agent's output.

        Args:
            context: Shared context dictionary with market data, signals, etc.

        Returns:
            Updated context dictionary with agent's contribution

        Example:
            >>> agent = DataSyncAgent(provider, config)
            >>> context = {"symbol": "BTC/USDT"}
            >>> result = await agent.execute(context)
            >>> "market_data" in result
            True
        """
        pass

    def get_agent_name(self) -> str:
        """Return agent name.

        Returns:
            Agent class name (e.g., "DataSyncAgent")
        """
        return self.__class__.__name__

    def log_decision(self, message: str, level: str = "info") -> None:
        """Log agent decision with consistent formatting.

        Args:
            message: Log message
            level: Log level ("info", "warning", "error")
        """
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"[{self.get_agent_name()}] {message}")
