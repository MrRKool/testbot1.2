#!/usr/bin/env python3
"""
Setup script for the trading bot.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)

def create_virtual_environment():
    """Create virtual environment."""
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            check=True,
        )
        logger.info("Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        sys.exit(1)

def install_requirements():
    """Install required packages."""
    try:
        # Upgrade pip
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )

        # Install requirements
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)

def install_ta_lib():
    """Install TA-Lib."""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["brew", "install", "ta-lib"], check=True)
        elif sys.platform == "linux":
            subprocess.run(
                [
                    "wget",
                    "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz",
                ],
                check=True,
            )
            subprocess.run(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"], check=True)
            os.chdir("ta-lib")
            subprocess.run(["./configure", "--prefix=/usr"], check=True)
            subprocess.run(["make"], check=True)
            subprocess.run(["sudo", "make", "install"], check=True)
            os.chdir("..")
            subprocess.run(["rm", "-rf", "ta-lib", "ta-lib-0.4.0-src.tar.gz"], check=True)
        else:
            logger.warning(
                "TA-Lib installation not supported for this platform. "
                "Please install it manually."
            )
            return

        logger.info("TA-Lib installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install TA-Lib: {e}")
        sys.exit(1)

def create_directories():
    """Create required directories."""
    directories = [
        "logs",
        "data",
        "reports",
        "results",
        "templates",
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Directories created successfully")

def setup_database():
    """Initialize database."""
    try:
        import sqlite3

        conn = sqlite3.connect("data/trading.db")
        cursor = conn.cursor()

        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position TEXT NOT NULL,
                size REAL NOT NULL,
                pnl REAL,
                commission REAL,
                slippage REAL,
                stop_loss REAL,
                take_profit REAL,
                strategy TEXT,
                indicators TEXT
            )
        """)

        # Create performance_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_return REAL,
                annual_return REAL
            )
        """)

        # Create market_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                indicators TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

def setup_config():
    """Create example configuration file."""
    try:
        if not Path("config/config.yaml").exists():
            Path("config").mkdir(exist_ok=True)
            with open("config/config.example.yaml", "r") as example:
                with open("config/config.yaml", "w") as config:
                    config.write(example.read())
            logger.info("Configuration file created successfully")
    except Exception as e:
        logger.error(f"Failed to create configuration file: {e}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup trading bot")
    parser.add_argument(
        "--skip-ta-lib",
        action="store_true",
        help="Skip TA-Lib installation",
    )
    return parser.parse_args()

def main():
    """Main function to run setup."""
    args = parse_args()

    # Check Python version
    check_python_version()

    # Create virtual environment
    create_virtual_environment()

    # Install requirements
    install_requirements()

    # Install TA-Lib
    if not args.skip_ta_lib:
        install_ta_lib()

    # Create directories
    create_directories()

    # Setup database
    setup_database()

    # Setup configuration
    setup_config()

    logger.info("Setup completed successfully")
    logger.info("Next steps:")
    logger.info("1. Activate the virtual environment:")
    logger.info("   - Linux/Mac: source venv/bin/activate")
    logger.info("   - Windows: venv\\Scripts\\activate")
    logger.info("2. Configure the bot in config/config.yaml")
    logger.info("3. Run the bot: python main.py")

if __name__ == "__main__":
    main() 