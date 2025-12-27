"""Base agent interface for trading agents.

All 8 agents in the adversarial decision system inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..config import TradingConfig
from ..providers.base import BaseExchangeProvider
from ..logging_config import get_logger, DecisionContext
from ..utils.timeout import with_timeout


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
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    @with_timeout(60.0)  # 60s timeout for agent execution
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute agent logic with timeout protection.

        This is the main entry point for each agent. Takes a context dictionary
        containing all available data and returns an updated context with the
        agent's output.

        Protected by 60s timeout to prevent infinite loops or hanging operations.

        Args:
            context: Shared context dictionary with market data, signals, etc.

        Returns:
            Updated context dictionary with agent's contribution

        Raises:
            AgentTimeoutError: If execution exceeds 60 seconds

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

    def log_decision(self, message: str, level: str = "info", **extra_data) -> None:
        """Log agent decision with structured logging and decision context.

        Args:
            message: Log message
            level: Log level ("info", "warning", "error")
            **extra_data: Additional context fields to include in log
        """
        log_method = getattr(self.logger, level, self.logger.info)

        # Merge decision context with extra data
        log_extra = {
            **DecisionContext.get_extra(),
            "agent": self.get_agent_name(),
            **extra_data,
        }

        log_method(message, extra=log_extra)
