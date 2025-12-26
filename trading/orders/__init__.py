"""Advanced Order Management - Bracket, Trailing, OCO orders.

This module provides professional order management capabilities including:
- Bracket orders (entry + stop loss + take profit)
- Trailing stop orders
- OCO (One-Cancels-Other) orders
- Order lifecycle tracking
- Risk-based position sizing

Components:
- OrderManager: Main interface for creating advanced orders
- OrderLifecycleManager: Tracks order state transitions
- BracketOrder: Entry with automatic risk management
- TrailingStopOrder: Dynamic stop-loss that trails price
- OCOOrder: Linked orders where one cancels the other

Example Usage:
    ```python
    from trading.orders import OrderManager, BracketOrder
    from trading.providers.factory import create_provider
    from trading.config import TradingConfig

    # Initialize order manager
    config = TradingConfig.from_env("binance_futures")
    provider = create_provider(config)
    order_manager = OrderManager(provider, config)

    # Create bracket order (entry + SL + TP)
    entry, stop_loss, take_profit = await order_manager.create_bracket_order(
        symbol="BTC/USDT",
        side="buy",
        amount=0.1,
        entry_price=42000,
        stop_loss_price=41000,  # Risk: $1000
        take_profit_price=44000,  # Reward: $2000
    )

    print(f"Entry: {entry.id}")
    print(f"Stop Loss: {stop_loss.id}")
    print(f"Take Profit: {take_profit.id}")

    # Create trailing stop
    trailing = await order_manager.create_trailing_stop(
        symbol="BTC/USDT",
        side="sell",
        amount=0.1,
        activation_price=42000,
        trailing_delta=2.0,  # 2% trail
        trailing_type=TrailingType.PERCENT,
    )

    # Update trailing stops on price changes
    await order_manager.update_trailing_stops({"BTC/USDT": 43000})

    # Calculate position size based on risk
    position_size = order_manager.calculate_position_size(
        entry_price=42000,
        stop_loss_price=41000,
        risk_amount_usd=100,  # Risk $100
    )
    print(f"Position size: {position_size:.4f} BTC")
    ```

Features:
- **Risk Management**: Automatic stop-loss and take-profit orders
- **Trailing Stops**: Dynamic stops that protect profits
- **OCO Orders**: Exit strategies with multiple outcomes
- **Position Sizing**: Calculate safe position sizes based on risk
- **Lifecycle Tracking**: Monitor order status and events
- **Parent-Child Links**: Automatic cancellation of related orders

Architecture:
The order management system sits between the execution agent and exchange
providers, translating high-level trading intentions into platform-specific
order sequences.

Data Flow:
1. ExecutionEngine requests bracket order via OrderManager
2. OrderManager creates entry, stop-loss, and take-profit orders
3. OrderLifecycleManager tracks all orders and relationships
4. When entry fills, stop-loss and take-profit are activated
5. When stop or take-profit fills, the other is automatically canceled

Risk Parameters:
- Default stop-loss: 2% of position value
- Default take-profit: 4% of position value (2:1 reward:risk)
- Minimum risk/reward ratio: 1.5:1
- Maximum position size: Configurable per account

Note:
- Not all exchanges support all order types
- OrderManager gracefully degrades to supported features
- Always test with paper trading before live deployment
"""

from .manager import OrderManager
from .lifecycle import OrderLifecycleManager, OrderEvent
from .types import (
    BracketOrder,
    TrailingStopOrder,
    OCOOrder,
    OrderGroup,
    AdvancedOrderType,
    TrailingType,
    OrderRelation,
)

__all__ = [
    # Core classes
    "OrderManager",
    "OrderLifecycleManager",
    "OrderEvent",

    # Order types
    "BracketOrder",
    "TrailingStopOrder",
    "OCOOrder",
    "OrderGroup",

    # Enums
    "AdvancedOrderType",
    "TrailingType",
    "OrderRelation",
]

__version__ = "1.0.0"

# Module metadata
__author__ = "LLM-TradeBot Contributors"
__description__ = "Advanced order management with bracket, trailing, and OCO orders"
