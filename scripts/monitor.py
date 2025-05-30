#!/usr/bin/env python3
"""
Script for monitoring the trading bot's performance and system health.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import requests
import yaml
from telegram import Bot

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("monitor", "logs/monitor.log")

class SystemMonitor:
    """Monitor system health."""

    def __init__(self, config):
        """Initialize system monitor.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.thresholds = config["monitoring"]["system_thresholds"]
        self.telegram_bot = None
        if config["monitoring"]["telegram"]["enabled"]:
            self.telegram_bot = Bot(token=config["monitoring"]["telegram"]["token"])

    def get_system_metrics(self):
        """Get system metrics.

        Returns:
            dict: System metrics
        """
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return None

    def check_system_health(self):
        """Check system health.

        Returns:
            bool: True if system is healthy
        """
        try:
            metrics = self.get_system_metrics()
            if metrics is None:
                return False

            # Check thresholds
            if metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
                self.send_alert(f"High CPU usage: {metrics['cpu_percent']}%")
                return False

            if metrics["memory_percent"] > self.thresholds["memory_percent"]:
                self.send_alert(f"High memory usage: {metrics['memory_percent']}%")
                return False

            if metrics["disk_percent"] > self.thresholds["disk_percent"]:
                self.send_alert(f"High disk usage: {metrics['disk_percent']}%")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return False

    def send_alert(self, message):
        """Send alert via Telegram.

        Args:
            message (str): Alert message
        """
        try:
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    chat_id=self.config["monitoring"]["telegram"]["chat_id"],
                    text=f"🚨 System Alert: {message}",
                )
            logger.warning(f"System Alert: {message}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

class TradingMonitor:
    """Monitor trading performance."""

    def __init__(self, config):
        """Initialize trading monitor.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.thresholds = config["monitoring"]["trading_thresholds"]
        self.telegram_bot = None
        if config["monitoring"]["telegram"]["enabled"]:
            self.telegram_bot = Bot(token=config["monitoring"]["telegram"]["token"])

    def check_exchange_health(self):
        """Check exchange health.

        Returns:
            bool: True if exchange is healthy
        """
        try:
            # Check exchange API
            response = requests.get(
                f"{self.config['exchange']['api']['base_url']}/ping",
                timeout=5,
            )
            if response.status_code != 200:
                self.send_alert("Exchange API is not responding")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking exchange health: {e}")
            self.send_alert("Error checking exchange health")
            return False

    def get_trading_metrics(self):
        """Get trading metrics from database.

        Returns:
            dict: Trading metrics
        """
        try:
            # Load recent trades
            trades = pd.read_sql(
                "SELECT * FROM trades WHERE exit_time IS NOT NULL",
                "sqlite:///data/trading.db",
                parse_dates=["entry_time", "exit_time"],
            )

            # Load performance metrics
            metrics = pd.read_sql(
                "SELECT * FROM performance_metrics",
                "sqlite:///data/trading.db",
                parse_dates=["timestamp"],
            )

            if trades.empty or metrics.empty:
                return None

            return {
                "win_rate": (trades["pnl"] > 0).mean(),
                "profit_factor": abs(trades[trades["pnl"] > 0]["pnl"].sum() / trades[trades["pnl"] < 0]["pnl"].sum()),
                "sharpe_ratio": metrics["sharpe_ratio"].iloc[-1],
                "max_drawdown": metrics["max_drawdown"].min(),
                "total_return": trades["pnl"].sum() / trades["size"].sum(),
                "annual_return": metrics["annual_return"].iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}")
            return None

    def check_trading_health(self):
        """Check trading health.

        Returns:
            bool: True if trading is healthy
        """
        try:
            metrics = self.get_trading_metrics()
            if metrics is None:
                return False

            # Check thresholds
            if metrics["win_rate"] < self.thresholds["min_win_rate"]:
                self.send_alert(f"Low win rate: {metrics['win_rate']:.2%}")
                return False

            if metrics["profit_factor"] < self.thresholds["min_profit_factor"]:
                self.send_alert(f"Low profit factor: {metrics['profit_factor']:.2f}")
                return False

            if metrics["sharpe_ratio"] < self.thresholds["min_sharpe_ratio"]:
                self.send_alert(f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                return False

            if metrics["max_drawdown"] < -self.thresholds["max_drawdown"]:
                self.send_alert(f"High drawdown: {metrics['max_drawdown']:.2%}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking trading health: {e}")
            return False

    def send_alert(self, message):
        """Send alert via Telegram.

        Args:
            message (str): Alert message
        """
        try:
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    chat_id=self.config["monitoring"]["telegram"]["chat_id"],
                    text=f"🚨 Trading Alert: {message}",
                )
            logger.warning(f"Trading Alert: {message}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor trading bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Monitoring interval in seconds",
    )
    return parser.parse_args()

def main():
    """Main function to run monitoring."""
    args = parse_args()

    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Initialize monitors
        system_monitor = SystemMonitor(config)
        trading_monitor = TradingMonitor(config)

        logger.info("Starting monitoring...")

        while True:
            # Check system health
            if not system_monitor.check_system_health():
                logger.error("System health check failed")

            # Check exchange health
            if not trading_monitor.check_exchange_health():
                logger.error("Exchange health check failed")

            # Check trading health
            if not trading_monitor.check_trading_health():
                logger.error("Trading health check failed")

            # Wait for next check
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        sys.exit(1)

        """Check trading health and return status."""
        metrics = self.get_trading_metrics()
        if not metrics:
            return "error", ["Failed to get trading metrics"]

        status = "healthy"
        alerts = []

        # Check win rate
        if metrics["win_rate"] < 0.4:
            status = "warning"
            alerts.append(f"Low win rate: {metrics['win_rate']:.2%}")

        # Check profit factor
        if metrics["profit_factor"] < 1.2:
            status = "warning"
            alerts.append(f"Low profit factor: {metrics['profit_factor']:.2f}")

        # Check drawdown
        if metrics["max_drawdown"] > 0.2:
            status = "warning"
            alerts.append(f"High drawdown: {metrics['max_drawdown']:.2%}")

        # Check Sharpe ratio
        if metrics["sharpe_ratio"] < 1.0:
            status = "warning"
            alerts.append(f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

        return status, alerts

    def monitor(self):
        """Run monitoring loop."""
        while True:
            try:
                # Check system health
                system_status, system_alerts = self.system_monitor.check_system_health()
                if system_status != "healthy":
                    alert_message = "⚠️ <b>System Alert</b>\n\n"
                    for alert in system_alerts:
                        alert_message += f"• {alert}\n"
                    self.system_monitor.send_telegram_alert(alert_message)

                # Check exchange health
                if not self.check_exchange_health():
                    self.system_monitor.send_telegram_alert(
                        "⚠️ <b>Exchange Alert</b>\n\n• Exchange API is not responding"
                    )

                # Check trading health
                trading_status, trading_alerts = self.check_trading_health()
                if trading_status != "healthy":
                    alert_message = "⚠️ <b>Trading Alert</b>\n\n"
                    for alert in trading_alerts:
                        alert_message += f"• {alert}\n"
                    self.system_monitor.send_telegram_alert(alert_message)

                # Get current metrics
                metrics = self.get_trading_metrics()
                if metrics:
                    self.metrics_history.append({
                        "timestamp": datetime.now(),
                        "metrics": metrics,
                    })

                    # Keep only last 24 hours of metrics
                    cutoff_time = datetime.now() - pd.Timedelta(hours=24)
                    self.metrics_history = [
                        m for m in self.metrics_history
                        if m["timestamp"] > cutoff_time
                    ]

                    # Save metrics to database
                    metrics_df = pd.DataFrame([{
                        "timestamp": datetime.now(),
                        **metrics,
                    }])
                    metrics_df.to_sql(
                        "performance_metrics",
                        f"sqlite:///{self.config['database']['path']}",
                        if_exists="append",
                        index=False,
                    )

                # Sleep until next check
                time.sleep(self.config["monitoring"]["metrics"]["update_interval"])

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self.system_monitor.send_telegram_alert(
                    f"❌ <b>Monitoring Error</b>\n\n• {str(e)}"
                )
                time.sleep(60)  # Wait a minute before retrying

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor trading bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()

def main():
    """Main function to run monitoring."""
    args = parse_args()
    config = load_config(args.config)

    # Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path(config["database"]["path"]).parent.mkdir(parents=True, exist_ok=True)

    # Initialize and run monitor
    monitor = TradingMonitor(config)
    monitor.monitor()

if __name__ == "__main__":
    main() 