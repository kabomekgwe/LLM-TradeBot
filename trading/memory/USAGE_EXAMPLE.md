# Trading Memory System - Usage Example

This document shows how to integrate the trade journal, pattern detection, and optional Graphiti memory into the trading system.

## Basic Trade Journaling

```python
from pathlib import Path
from integrations.trading.memory.trade_history import TradeJournal, TradeRecord
from integrations.trading.models.positions import Order, OrderSide, OrderType, OrderStatus

# Initialize journal for a spec
spec_dir = Path("specs/001-trading-feature")
journal = TradeJournal(spec_dir)

# Log a trade (typically done after order execution)
trade = TradeRecord.from_order(order, decision=trading_decision)
journal.log_trade(trade)

# When trade is closed, update it
trade.close_trade(exit_price=42500.0, fees=5.0)
journal.log_trade(trade)  # Re-save with closure info

# Get recent trades
recent_trades = journal.get_recent_trades(limit=10)

# Calculate performance metrics
metrics = journal.calculate_metrics()
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Total P&L: ${metrics['total_pnl']:.2f}")
```

## Pattern Detection & Insights

```python
from integrations.trading.memory.patterns import PatternDetector

# Initialize pattern detector
detector = PatternDetector(journal)

# Analyze regime performance
regime_insights = detector.analyze_regime_performance()
for insight in regime_insights:
    print(f"{insight.title} (Confidence: {insight.confidence * 100:.0f}%)")
    print(insight.description)

# Analyze agent behavior
agent_insights = detector.analyze_agent_behavior()

# Generate all insights and save to markdown
all_insights = detector.generate_all_insights()
detector.save_insights_to_markdown(spec_dir / "memory" / "trading_insights.md")
```

## Integration with TradingManager

```python
from integrations.trading.manager import TradingManager
from integrations.trading.memory.trade_history import TradeJournal, TradeRecord

class EnhancedTradingManager(TradingManager):
    """Trading manager with memory integration."""

    def __init__(self, config, spec_dir):
        super().__init__(config)
        self.spec_dir = spec_dir
        self.journal = TradeJournal(spec_dir)

    async def run_trading_loop(self, symbol: str = "BTC/USDT") -> dict:
        """Run trading loop with trade journaling."""

        # Execute standard trading loop
        result = await super().run_trading_loop(symbol)

        # If order was created, log it
        if result.get("success") and result.get("order"):
            order = result["order"]
            decision = result.get("decision")

            # Create trade record
            trade = TradeRecord.from_order(order, decision)
            self.journal.log_trade(trade)

            # Store trade_id for later closure
            result["trade_id"] = trade.trade_id

        return result

    async def close_position(self, symbol: str) -> dict:
        """Close position and update trade journal."""

        # Execute closure
        result = await super().close_position(symbol)

        if result.get("success") and result.get("order"):
            # Find the original trade and update it
            # (In real implementation, you'd track trade_id properly)
            exit_order = result["order"]

            # Update trade journal with closure
            # trade.close_trade(exit_price=exit_order.price, fees=...)
            # self.journal.log_trade(trade)

        return result

    def generate_insights(self) -> list:
        """Generate trading insights from history."""
        from integrations.trading.memory.patterns import PatternDetector

        detector = PatternDetector(self.journal)
        insights = detector.generate_all_insights()

        # Save to markdown
        insights_file = self.spec_dir / "memory" / "trading_insights.md"
        detector.save_insights_to_markdown(insights_file)

        return insights
```

## Optional Graphiti Integration

```python
from integrations.trading.memory.graphiti_trading import TradingMemory

# Initialize Graphiti memory (requires GRAPHITI_ENABLED=true in .env)
memory = TradingMemory(spec_name="trading-001")

if memory.is_available():
    # Save trade as episode
    await memory.save_trade_episode(trade, decision)

    # Find similar market conditions
    similar = await memory.find_similar_market_conditions(current_decision, limit=5)
    for episode in similar:
        print(f"Similar scenario (relevance: {episode['relevance']:.2f})")
        print(episode['narrative'])

    # Get contextual recommendation
    recommendation = await memory.get_contextual_recommendation(current_decision)
    if recommendation:
        print(f"Recommendation: {recommendation}")

    # Learn from past mistakes
    insights = await memory.learn_from_mistakes()
    for insight in insights:
        print(f"Learning: {insight}")
else:
    print("Graphiti not available - using file-based memory only")
```

## Automatic Insights Generation

The system can automatically generate insights after each trading session:

```python
# At the end of a trading session
manager = EnhancedTradingManager(config, spec_dir)

# Run some trades...
await manager.run_trading_loop("BTC/USDT")
await manager.run_trading_loop("ETH/USDT")

# Generate insights
insights = manager.generate_insights()

# Insights are saved to: specs/XXX/memory/trading_insights.md
# Example output:
# - "Strong Performance in TRENDING Markets (Confidence: 85%)"
# - "Bull Agent Shows Strong Predictive Accuracy (Confidence: 72%)"
# - "High Confidence Trades Significantly Outperform (Confidence: 90%)"
```

## CLI Usage

```bash
# Generate insights from trade history
python integrations/trading/cli.py insights

# Output will include:
# - insights.json (structured data)
# - specs/XXX/memory/trading_insights.md (human-readable)
```

## Directory Structure

After using the memory system, your spec directory will look like:

```
specs/001-trading-feature/
├── spec.md
├── requirements.json
├── .trading_state.json
└── memory/
    ├── trades/
    │   ├── index.json
    │   ├── 20250124_143022_paper_1.json
    │   ├── 20250124_143045_paper_2.json
    │   └── 20250124_143112_paper_3.json
    └── trading_insights.md
```

## Key Features

1. **File-Based Journaling**: All trades stored as human-readable JSON
2. **Performance Metrics**: Automatic calculation of win rate, P&L, profit factor, etc.
3. **Pattern Detection**: Analyzes performance by regime, symbol, confidence level
4. **Insights Generation**: Automatically identifies what's working and what's not
5. **Optional Graphiti**: Semantic search and cross-session learning when enabled
6. **Graceful Degradation**: Works perfectly fine without Graphiti

## Best Practices

1. **Log Every Trade**: Always call `journal.log_trade()` after order execution
2. **Update on Closure**: Call `trade.close_trade()` and re-log when position closes
3. **Regular Insights**: Generate insights weekly to track performance trends
4. **Review Patterns**: Actually read the generated `trading_insights.md` file
5. **Act on Learnings**: Adjust strategy based on discovered patterns
