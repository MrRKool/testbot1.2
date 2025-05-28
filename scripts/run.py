#!/usr/bin/env python3
"""
Script for running the trading bot.
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.bot import TradingBot
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("run", "logs/run.log")

def signal_handler(signum, frame):
    """Handle shutdown signals.

    Args:
        signum (int): Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, "bot"):
        signal_handler.bot.stop()
    sys.exit(0)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trading bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Trading symbols to use",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization before starting",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable performance monitoring",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report",
    )
    return parser.parse_args()

def main():
    """Main function to run the trading bot."""
    args = parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize trading bot
        bot = TradingBot(
            config_path=args.config,
            symbols=args.symbols,
            backtest=args.backtest,
            optimize=args.optimize,
            monitor=args.monitor,
            report=args.report,
        )

        # Store bot instance for signal handler
        signal_handler.bot = bot

        # Start the bot
        logger.info("Starting trading bot...")
        bot.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        if "bot" in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 