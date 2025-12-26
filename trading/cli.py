#!/usr/bin/env python3
"""Trading integration CLI for IPC communication.

This CLI provides JSON-based commands for the Electron UI to interact
with the trading system via IPC handlers.
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.trading.config import TradingConfig
from integrations.trading.manager import TradingManager
from integrations.trading.providers.factory import create_provider


def json_output(data: dict):
    """Print JSON output for IPC consumption."""
    print(json.dumps(data, indent=2))


async def cmd_status():
    """Get trading system status and configuration."""
    try:
        config = TradingConfig.from_env()

        # Load state if exists
        spec_dir = Path.cwd() / "specs"
        state_files = list(spec_dir.glob("*/.trading_state.json"))

        state_data = None
        if state_files:
            from integrations.trading.state import TradingState
            state = TradingState.load(state_files[0].parent)
            state_data = {
                "initialized": state.initialized,
                "total_trades": state.total_trades,
                "win_rate": state.win_rate,
                "total_pnl": state.total_pnl,
                "circuit_breaker_tripped": state.circuit_breaker_tripped,
                "active_positions": len(state.active_positions),
            }

        json_output({
            "success": True,
            "config": {
                "provider": config.provider,
                "testnet": config.testnet,
                "max_position_size_usd": config.max_position_size_usd,
                "max_daily_drawdown_pct": config.max_daily_drawdown_pct,
                "max_open_positions": config.max_open_positions,
                "decision_threshold": config.decision_threshold,
            },
            "state": state_data,
        })

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_positions():
    """Get current open positions and balance."""
    try:
        config = TradingConfig.from_env()
        provider = create_provider(config)

        try:
            positions_data = await provider.fetch_positions()
            balance_data = await provider.fetch_balance()

            # Convert to dict format
            positions = [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage,
                    "pnl_pct": p.pnl_pct,
                    "is_profitable": p.is_profitable,
                }
                for p in positions_data
            ]

            balance = {
                "currency": balance_data.currency,
                "free": balance_data.free,
                "used": balance_data.used,
                "total": balance_data.total,
            }

            json_output({
                "success": True,
                "positions": positions,
                "balance": balance,
            })

        finally:
            await provider.close()

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_run(args):
    """Execute a full trading loop."""
    try:
        config = TradingConfig.from_env()

        # TODO: Initialize TradingManager and run trading loop
        # For now, return a placeholder
        json_output({
            "success": True,
            "message": "Trading loop execution coming in Phase 1 completion",
            "dry_run": args.dry_run,
            "symbol": args.symbol,
        })

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_history(args):
    """Get trading history."""
    try:
        # Load state file
        spec_dir = Path.cwd() / "specs"
        state_files = list(spec_dir.glob("*/.trading_state.json"))

        if not state_files:
            json_output({
                "success": True,
                "trades": [],
                "total_pnl": 0.0,
                "win_rate": 0.0,
            })
            return

        from integrations.trading.state import TradingState
        state = TradingState.load(state_files[0].parent)

        json_output({
            "success": True,
            "trades": [],  # TODO: Implement trade history storage
            "total_pnl": state.total_pnl,
            "win_rate": state.win_rate,
        })

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_cancel(args):
    """Cancel an order."""
    try:
        config = TradingConfig.from_env()
        provider = create_provider(config)

        try:
            success = await provider.cancel_order(args.order_id, args.symbol)

            json_output({
                "success": True,
                "canceled": success,
            })

        finally:
            await provider.close()

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_close(args):
    """Close a position."""
    try:
        config = TradingConfig.from_env()
        provider = create_provider(config)

        try:
            # Get current position
            positions = await provider.fetch_positions()
            position = next((p for p in positions if p.symbol == args.symbol), None)

            if not position:
                json_output({
                    "success": False,
                    "error": f"No open position for {args.symbol}",
                })
                return

            # Create opposite order to close
            close_side = "sell" if position.side.value == "long" else "buy"
            order = await provider.create_order(
                symbol=args.symbol,
                side=close_side,
                order_type="market",
                amount=position.size,
            )

            json_output({
                "success": True,
                "closed": order.status.value == "filled",
                "order_id": order.id,
            })

        finally:
            await provider.close()

    except Exception as e:
        json_output({"success": False, "error": str(e)})


async def cmd_insights():
    """Generate trading insights from historical data."""
    try:
        from integrations.trading.memory.trade_history import TradeJournal
        from integrations.trading.memory.patterns import PatternDetector

        # Find spec directory
        spec_dir = Path.cwd() / "specs"
        spec_dirs = [d for d in spec_dir.glob("*") if d.is_dir()]

        if not spec_dirs:
            json_output({
                "success": False,
                "error": "No spec directories found",
            })
            return

        # Use most recent spec
        latest_spec = max(spec_dirs, key=lambda x: x.stat().st_mtime)

        # Load trade journal
        journal = TradeJournal(latest_spec)
        detector = PatternDetector(journal)

        # Generate insights
        insights = detector.generate_all_insights()

        # Save to markdown
        insights_file = latest_spec / "memory" / "trading_insights.md"
        detector.save_insights_to_markdown(insights_file)

        # Return insights
        json_output({
            "success": True,
            "insights": [
                {
                    "category": i.category,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "sample_size": i.sample_size,
                }
                for i in insights
            ],
            "insights_file": str(insights_file),
        })

    except Exception as e:
        json_output({"success": False, "error": str(e)})


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trading integration CLI for IPC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status command
    subparsers.add_parser("status", help="Get trading system status")

    # Positions command
    subparsers.add_parser("positions", help="Get current positions and balance")

    # Run trading loop command
    run_parser = subparsers.add_parser("run", help="Execute trading loop")
    run_parser.add_argument("--symbol", required=True, help="Trading symbol")
    run_parser.add_argument("--dry-run", action="store_true", help="Dry run mode")

    # History command
    history_parser = subparsers.add_parser("history", help="Get trading history")
    history_parser.add_argument("--limit", type=int, default=50, help="Limit results")

    # Cancel order command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel an order")
    cancel_parser.add_argument("--order-id", required=True, help="Order ID")
    cancel_parser.add_argument("--symbol", required=True, help="Trading symbol")

    # Close position command
    close_parser = subparsers.add_parser("close", help="Close a position")
    close_parser.add_argument("--symbol", required=True, help="Trading symbol")

    # Insights command
    subparsers.add_parser("insights", help="Generate trading insights from history")

    args = parser.parse_args()

    # Execute command
    if args.command == "status":
        asyncio.run(cmd_status())
    elif args.command == "positions":
        asyncio.run(cmd_positions())
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "history":
        asyncio.run(cmd_history(args))
    elif args.command == "cancel":
        asyncio.run(cmd_cancel(args))
    elif args.command == "close":
        asyncio.run(cmd_close(args))
    elif args.command == "insights":
        asyncio.run(cmd_insights())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
