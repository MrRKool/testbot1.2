import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import traceback
from pathlib import Path
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
import asyncio
import sqlite3
import json
import atexit
from utils.trading.price_fetcher import PriceFetcher, PriceFetcherConfig
from utils.trading.strategy import Strategy
from utils.trading.trade_executor import TradeExecutor, Trade, OrderType, OrderStatus, TradeConfig
from utils.telegram_alerts import TelegramAlert, TelegramConfig, AlertType
from utils.dependency_checker import check_dependencies
from utils.risk_utils import get_risk_param
from utils.monitoring.performance_monitor import PerformanceMonitor
from utils.database.database_manager import DatabaseManager
from utils.cache.cache_manager import CacheManager
from utils.rate_limiter import RateLimiter
from utils.config import load_config
from utils.voice.voice_controller import VoiceController
from utils.system_service import ServiceManager
import psutil
from src.ai.ai_strategy import AITradingStrategy
from src.backtest.backtest_engine import BacktestEngine
import matplotlib.pyplot as plt
import argparse
import gc
from functools import lru_cache
from contextlib import contextmanager
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import threading

# Import the new centralized logger
from utils.logger import logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    logger.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))

# Set the global exception handler
sys.excepthook = handle_exception

def main():
    """Main entry point for the trading bot."""
    try:
        # Initialize logger
        logger.info("Starting trading bot...")
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check dependencies
        check_dependencies()
        logger.info("Dependencies checked successfully")
        
        # Initialize components
        try:
            # Initialize database
            db_manager = DatabaseManager()
            logger.info("Database initialized successfully")
            
            # Initialize cache
            cache_manager = CacheManager()
            logger.info("Cache initialized successfully")
            
            # Initialize price fetcher
            price_fetcher = PriceFetcher(PriceFetcherConfig())
            logger.info("Price fetcher initialized successfully")
            
            # Initialize strategy
            strategy = Strategy()
            logger.info("Strategy initialized successfully")
            
            # Initialize trade executor
            trade_executor = TradeExecutor(TradeConfig())
            logger.info("Trade executor initialized successfully")
            
            # Initialize performance monitor
            performance_monitor = PerformanceMonitor()
            logger.info("Performance monitor initialized successfully")
            
            # Initialize AI strategy
            ai_strategy = AITradingStrategy()
            logger.info("AI strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            raise
        
        # Start the main trading loop
        logger.info("Starting main trading loop...")
        while True:
            try:
                # Your trading logic here
                pass
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 