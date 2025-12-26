"""
Basic Trading Example
=====================

This example shows how to:
1. Initialize the trading system
2. Run a trading decision loop
3. View positions and balance
4. Analyze trade history
"""

import asyncio
from pathlib import Path
from trading.config import TradingConfig
from trading.providers.factory import create_provider
from trading.memory import TradeJournal, PatternDetector


async def basic_trading_example():
    """Run a basic trading example with paper trading."""

    print("🤖 LLM-TradeBot - Basic Trading Example\n")

    # 1. Load configuration from .env file
    print("1️⃣ Loading configuration...")
    config = TradingConfig.from_env()
    print(f"   Provider: {config.provider}")
    print(f"   Testnet: {config.testnet}")
    print(f"   Max Position Size: ${config.max_position_size_usd}\n")

    # 2. Create provider
    print("2️⃣ Initializing trading provider...")
    provider = create_provider(config)
    print(f"   ✓ {provider.get_provider_name()} provider ready\n")

    try:
        # 3. Fetch current balance
        print("3️⃣ Fetching account balance...")
        balance = await provider.fetch_balance()
        print(f"   Currency: {balance.currency}")
        print(f"   Total: {balance.total:.2f}")
        print(f"   Free: {balance.free:.2f}")
        print(f"   Used: {balance.used:.2f}\n")

        # 4. Fetch current positions
        print("4️⃣ Checking open positions...")
        positions = await provider.fetch_positions()
        if positions:
            print(f"   Found {len(positions)} open position(s):")
            for pos in positions:
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                print(f"   - {pos.symbol}: {pos.side.value} {pos.size} @ ${pos.entry_price:.2f}")
                print(f"     P&L: {pnl_sign}${pos.unrealized_pnl:.2f} ({pos.pnl_pct:.2f}%)")
        else:
            print("   No open positions\n")

        # 5. Fetch market data
        print("5️⃣ Fetching market data for BTC/USDT...")
        ticker = await provider.fetch_ticker("BTC/USDT")
        print(f"   Bid: ${ticker.bid:.2f}")
        print(f"   Ask: ${ticker.ask:.2f}")
        print(f"   Last: ${ticker.last:.2f}")
        print(f"   Spread: ${ticker.spread:.2f}\n")

        # 6. Example: Create a small test order (paper trading only!)
        if config.provider == "paper":
            print("6️⃣ Creating test order (paper trading)...")
            order = await provider.create_order(
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                amount=0.001,  # Small test amount
            )
            print(f"   ✓ Order created: {order.id}")
            print(f"   Status: {order.status.value}")
            print(f"   Filled: {order.filled}/{order.amount}\n")
        else:
            print("6️⃣ Skipping test order (not in paper trading mode)\n")

    finally:
        # Always close provider connection
        await provider.close()
        print("✓ Provider connection closed\n")


async def analyze_trade_history():
    """Analyze trade history and generate insights."""

    print("📊 Trade History Analysis\n")

    # Setup journal (you'll need a spec directory)
    data_dir = Path("data/trading")
    data_dir.mkdir(parents=True, exist_ok=True)

    journal = TradeJournal(data_dir)

    # Get metrics
    metrics = journal.calculate_metrics()
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Closed Trades: {metrics['closed_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}\n")

    # Generate insights
    if metrics['total_trades'] >= 10:
        print("Generating insights...\n")
        detector = PatternDetector(journal)
        insights = detector.generate_all_insights()

        for insight in insights[:3]:  # Show top 3
            print(f"💡 {insight.title}")
            print(f"   Confidence: {insight.confidence * 100:.0f}%")
            print(f"   {insight.description[:100]}...\n")
    else:
        print("Not enough trades for pattern analysis (need 10+)\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("LLM-TradeBot Example")
    print("=" * 60)
    print()

    # Run basic trading example
    asyncio.run(basic_trading_example())

    print("=" * 60)
    print()

    # Analyze trade history
    asyncio.run(analyze_trade_history())

    print("=" * 60)
    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("1. Check your .env configuration")
    print("2. Run: python -m trading.cli status")
    print("3. Try: python -m trading.cli run --symbol BTC/USDT")
    print()


if __name__ == "__main__":
    main()
