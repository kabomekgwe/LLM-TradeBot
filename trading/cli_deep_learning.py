#!/usr/bin/env python3
"""Independent CLI for Deep Learning Trading Strategy.

WHY separate CLI (from 07-03-PLAN.md):
- Independent execution: Can run alongside existing ensemble CLI
- Model selection: Choose best performer (BiLSTM or Transformer)
- Clear separation: No confusion with ensemble system

Usage:
    python trading/cli_deep_learning.py --model lstm
    python trading/cli_deep_learning.py --model transformer
"""

import asyncio
import argparse
import sys
from pathlib import Path

from trading.config import TradingConfig
from trading.providers.factory import create_provider
from trading.ml.deep_learning import DeepLearningStrategy
from trading.logging_config import get_logger


logger = get_logger(__name__)


async def run_strategy(model_type: str, symbol: str = "BTC/USDT"):
    """Run deep learning trading strategy.

    Args:
        model_type: "lstm" or "transformer"
        symbol: Trading pair (default: BTC/USDT)
    """
    logger.info(f"Starting Deep Learning Strategy with {model_type} model")

    try:
        # Load configuration from environment
        config = TradingConfig.from_env()

        if not config.is_valid():
            logger.error("Invalid trading configuration. Check environment variables.")
            return

        # Initialize exchange provider
        logger.info(f"Initializing {config.provider} provider...")
        provider = create_provider(config)

        # Initialize deep learning strategy
        logger.info(f"Initializing DeepLearningStrategy with {model_type} model...")
        strategy = DeepLearningStrategy(
            config=config,
            provider=provider,
            model_type=model_type,
            spec_dir=Path(f"specs/deep_learning_{model_type}")
        )

        # Run strategy execution loop
        logger.info(f"Running strategy for {symbol}...")
        while True:
            try:
                result = await strategy.execute_strategy(symbol)

                if result:
                    logger.info(
                        f"Trade executed: {result['signal']} "
                        f"(confidence={result['confidence']:.3f})"
                    )
                else:
                    logger.info("No trade executed this cycle")

                # Wait before next iteration (e.g., 5 minutes)
                logger.debug("Waiting 5 minutes before next cycle...")
                await asyncio.sleep(300)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for deep learning CLI."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Trading Strategy CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with BiLSTM model
  python trading/cli_deep_learning.py --model lstm

  # Run with Transformer model
  python trading/cli_deep_learning.py --model transformer

  # Run with specific trading pair
  python trading/cli_deep_learning.py --model lstm --symbol ETH/USDT
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lstm', 'transformer'],
        help='Model type to use (lstm or transformer)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair symbol (default: BTC/USDT)'
    )

    args = parser.parse_args()

    # Run async strategy
    try:
        asyncio.run(run_strategy(args.model, args.symbol))
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)


if __name__ == '__main__':
    main()
