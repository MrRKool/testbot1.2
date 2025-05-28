#!/usr/bin/env python3
"""
Script for running backtests.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.bot import TradingBot
from src.utils.logger import setup_logger
from src.utils.performance_analyzer import PerformanceAnalyzer

# Configure logging
logger = setup_logger("backtest", "logs/backtest.log")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Trading symbols to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/backtest",
        help="Output directory for results",
    )
    return parser.parse_args()

def validate_dates(start_date, end_date):
    """Validate and parse date strings.

    Args:
        start_date (str): Start date string
        end_date (str): End date string

    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("Start date must be before end date")
        return start, end
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

def main():
    """Main function to run backtest."""
    args = parse_args()

    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate dates
        start_date, end_date = validate_dates(args.start_date, args.end_date)

        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Initialize bot and analyzer
        bot = TradingBot(config_path=args.config, symbols=args.symbols)
        analyzer = PerformanceAnalyzer(config)

        # Run backtest for each symbol
        results = {}
        for symbol in tqdm(args.symbols or config["trading"]["symbols"], desc="Running backtests"):
            logger.info(f"Running backtest for {symbol}")

            # Run backtest
            trades = bot.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            # Analyze results
            metrics = analyzer.analyze_trades(trades)

            # Store results
            results[symbol] = {
                "trades": trades,
                "metrics": metrics,
            }

            # Save trades to CSV
            pd.DataFrame(trades).to_csv(
                output_dir / f"{symbol}_trades.csv",
                index=False,
            )

            # Save metrics to YAML
            with open(output_dir / f"{symbol}_metrics.yaml", "w") as f:
                yaml.dump(metrics, f)

        # Generate summary report
        summary = "Backtest Summary\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Period: {start_date.date()} to {end_date.date()}\n\n"

        for symbol, result in results.items():
            summary += f"Symbol: {symbol}\n"
            summary += "-" * 30 + "\n"
            summary += f"Total Trades: {len(result['trades'])}\n"
            summary += f"Win Rate: {result['metrics']['win_rate']:.2%}\n"
            summary += f"Profit Factor: {result['metrics']['profit_factor']:.2f}\n"
            summary += f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}\n"
            summary += f"Max Drawdown: {result['metrics']['max_drawdown']:.2%}\n"
            summary += f"Total Return: {result['metrics']['total_return']:.2%}\n"
            summary += f"Annual Return: {result['metrics']['annual_return']:.2%}\n"
            summary += "\n"

        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write(summary)

        logger.info(f"Backtest completed successfully")
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 