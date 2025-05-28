#!/usr/bin/env python3
"""
Script for generating trading performance reports.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from jinja2 import Environment, FileSystemLoader

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.utils.performance_analyzer import PerformanceAnalyzer

# Configure logging
logger = setup_logger("report", "logs/report.log")

class ReportGenerator:
    """Generator for trading performance reports."""

    def __init__(self, config):
        """Initialize report generator.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.analyzer = PerformanceAnalyzer(config)
        self.env = Environment(
            loader=FileSystemLoader(project_root / "templates")
        )

    def load_trading_data(self, start_date=None, end_date=None):
        """Load trading data from database.

        Args:
            start_date (str): Start date for filtering
            end_date (str): End date for filtering

        Returns:
            tuple: (trades, metrics)
        """
        try:
            # Load trades
            trades = pd.read_sql(
                "SELECT * FROM trades",
                "sqlite:///data/trading.db",
                parse_dates=["entry_time", "exit_time"],
            )

            # Filter by date if specified
            if start_date:
                trades = trades[trades["entry_time"] >= start_date]
            if end_date:
                trades = trades[trades["entry_time"] <= end_date]

            # Load performance metrics
            metrics = pd.read_sql(
                "SELECT * FROM performance_metrics",
                "sqlite:///data/trading.db",
                parse_dates=["timestamp"],
            )

            # Filter by date if specified
            if start_date:
                metrics = metrics[metrics["timestamp"] >= start_date]
            if end_date:
                metrics = metrics[metrics["timestamp"] <= end_date]

            return trades, metrics

        except Exception as e:
            logger.error(f"Error loading trading data: {e}")
            sys.exit(1)

    def generate_equity_curve(self, trades, output_dir):
        """Generate equity curve plot.

        Args:
            trades (pd.DataFrame): Trades data
            output_dir (Path): Output directory
        """
        try:
            # Calculate cumulative returns
            trades["return"] = trades["pnl"] / trades["size"]
            cumulative_returns = trades["return"].cumsum()

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative_returns.index, cumulative_returns.values)
            plt.title("Equity Curve")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative Return")
            plt.grid(True)

            # Save plot
            plt.savefig(output_dir / "equity_curve.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error generating equity curve: {e}")

    def generate_drawdown_plot(self, trades, output_dir):
        """Generate drawdown plot.

        Args:
            trades (pd.DataFrame): Trades data
            output_dir (Path): Output directory
        """
        try:
            # Calculate drawdown
            trades["return"] = trades["pnl"] / trades["size"]
            cumulative_returns = trades["return"].cumsum()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(drawdown.index, drawdown.values)
            plt.title("Drawdown")
            plt.xlabel("Trade Number")
            plt.ylabel("Drawdown")
            plt.grid(True)

            # Save plot
            plt.savefig(output_dir / "drawdown.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error generating drawdown plot: {e}")

    def generate_win_rate_plot(self, trades, output_dir):
        """Generate win rate plot.

        Args:
            trades (pd.DataFrame): Trades data
            output_dir (Path): Output directory
        """
        try:
            # Calculate win rate
            trades["win"] = trades["pnl"] > 0
            win_rate = trades["win"].rolling(window=20).mean()

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(win_rate.index, win_rate.values)
            plt.title("Win Rate (20-trade moving average)")
            plt.xlabel("Trade Number")
            plt.ylabel("Win Rate")
            plt.grid(True)

            # Save plot
            plt.savefig(output_dir / "win_rate.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error generating win rate plot: {e}")

    def generate_metrics_plot(self, metrics, output_dir):
        """Generate performance metrics plot.

        Args:
            metrics (pd.DataFrame): Performance metrics data
            output_dir (Path): Output directory
        """
        try:
            # Create plot
            plt.figure(figsize=(12, 6))
            for column in ["sharpe_ratio", "profit_factor"]:
                plt.plot(metrics["timestamp"], metrics[column], label=column)
            plt.title("Performance Metrics")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig(output_dir / "metrics.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error generating metrics plot: {e}")

    def generate_html_report(self, trades, metrics, output_dir):
        """Generate HTML report.

        Args:
            trades (pd.DataFrame): Trades data
            metrics (pd.DataFrame): Performance metrics data
            output_dir (Path): Output directory
        """
        try:
            # Calculate summary metrics
            summary = {
                "total_trades": len(trades),
                "win_rate": (trades["pnl"] > 0).mean(),
                "profit_factor": abs(trades[trades["pnl"] > 0]["pnl"].sum() / trades[trades["pnl"] < 0]["pnl"].sum()),
                "sharpe_ratio": metrics["sharpe_ratio"].iloc[-1],
                "max_drawdown": metrics["max_drawdown"].min(),
                "total_return": trades["pnl"].sum() / trades["size"].sum(),
                "annual_return": metrics["annual_return"].iloc[-1],
            }

            # Load template
            template = self.env.get_template("report.html")

            # Render template
            html = template.render(
                summary=summary,
                plots={
                    "equity_curve": "equity_curve.png",
                    "drawdown": "drawdown.png",
                    "win_rate": "win_rate.png",
                    "metrics": "metrics.png",
                },
            )

            # Save report
            with open(output_dir / "report.html", "w") as f:
                f.write(html)

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")

    def generate_report(self, start_date=None, end_date=None, output_dir=None):
        """Generate performance report.

        Args:
            start_date (str): Start date for filtering
            end_date (str): End date for filtering
            output_dir (str): Output directory
        """
        try:
            # Create output directory
            output_dir = Path(output_dir or "reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load trading data
            trades, metrics = self.load_trading_data(start_date, end_date)

            # Generate plots
            self.generate_equity_curve(trades, output_dir)
            self.generate_drawdown_plot(trades, output_dir)
            self.generate_win_rate_plot(trades, output_dir)
            self.generate_metrics_plot(metrics, output_dir)

            # Generate HTML report
            self.generate_html_report(trades, metrics, output_dir)

            logger.info(f"Report generated successfully in {output_dir}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate trading performance report")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for filtering (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for filtering (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for report",
    )
    return parser.parse_args()

def main():
    """Main function to generate report."""
    args = parse_args()

    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Initialize report generator
        generator = ReportGenerator(config)

        # Generate report
        generator.generate_report(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output,
        )

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 