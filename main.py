import os
import sys
import time
import signal
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import traceback
from pathlib import Path
import pandas as pd
import talib
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

# Eigen modules
from utils.logger import Logger, LogConfig, setup_logging
from utils.env_loader import check_environment, get_api_keys, get_telegram_config

# Configureer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log')
    ]
)

logger = logging.getLogger(__name__)

class TradingBot:
    """Hoofdclass voor de trading bot."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialiseer de trading bot."""
        try:
            logger.debug("Initialiseren TradingBot...")
            self.logger = setup_logging()
            self.log = self.logger.get_logger(__name__)
            
            # Basis configuratie
            self.config = {
                'exchange': {
                    'name': 'bybit',
                    'testnet': True,
                    'api': {
                        'base_url': 'https://api-testnet.bybit.com',
                        'timeout': 10,
                        'max_retries': 3,
                        'retry_delay': 5,
                        'rate_limit': {
                            'calls': 2,
                            'period': 1
                        }
                    }
                },
                'trading': {
                    'cycle_interval': 60,
                    'timeframes': ['1m', '5m', '15m', '1h'],
                    'min_volume': 1000000,
                    'volume_ma_period': 20,
                    'min_atr': 0.5,
                    'atr_period': 14,
                    'max_spread': 0.001,
                    'min_margin': 0.1,
                    'max_margin': 0.9,
                    'min_risk_reward': 2.0,
                    'max_open_trades': 3,
                    'max_daily_trades': 10,
                    'max_daily_loss': 0.02,
                    'symbols': {
                        'BTCUSDT': {
                            'min_volume': 1000000,
                            'min_volatility': 0.001,
                            'max_spread': 0.001,
                            'leverage': 1,
                            'enabled': True
                        }
                    }
                },
                'strategy': {
                    'indicators': {
                        'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
                        'ema': {'short_period': 9, 'medium_period': 21, 'long_period': 50},
                        'volume_ma': {'period': 20},
                        'atr': {'period': 14},
                        'adx': {'period': 14, 'threshold': 25},
                        'stochastic_rsi': {'rsi_period': 14, 'stochastic_period': 14, 'k_period': 3, 'd_period': 3},
                        'obv': {'smoothing': 5}
                    },
                    'market_regime': {
                        'min_confidence': 0.6,
                        'lookback_period': 100
                    },
                    'signal': {
                        'min_signals': 3,
                        'min_strength': 0.6,
                        'min_volume_multiplier': 1.5,
                        'min_atr_multiplier': 1.2
                    }
                },
                'risk': {
                    'max_position_size': 0.02,
                    'max_daily_trades': 10,
                    'max_daily_loss': 0.02,
                    'max_drawdown': 0.15,
                    'min_risk_reward': 2.0,
                    'max_open_trades': 3,
                    'base_risk_per_trade': 0.01,
                    'stop_loss': {
                        'type': 'fixed',
                        'value': 0.015
                    },
                    'take_profit': {
                        'type': 'fixed',
                        'value': 0.03
                    },
                    'trailing_stop': {
                        'enabled': True,
                        'activation': 0.01,
                        'distance': 0.005
                    },
                    'max_trade_duration': 24
                },
                'performance': {
                    'enabled': True,
                    'metrics': [
                        'win_rate',
                        'profit_factor',
                        'sharpe_ratio',
                        'max_drawdown',
                        'average_win',
                        'average_loss'
                    ],
                    'report_interval': 86400,
                    'save_reports': True,
                    'report_dir': 'reports'
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'date_format': '%Y-%m-%d %H:%M:%S',
                    'file': {
                        'enabled': True,
                        'path': 'logs/trading_bot.log',
                        'max_size': 10485760,
                        'backup_count': 5
                    },
                    'console': {
                        'enabled': True
                    }
                }
            }
            
            # Database initialisatie
            self.db_path = "trading_bot.db"
            self._init_database()
            
            # Performance monitor initialisatie
            self.performance_monitor = PerformanceMonitor(
                report_dir=self.config['performance']['report_dir'],
                metrics_window=1000,
                report_interval=self.config['performance']['report_interval']
            )
            
            # Laad API keys en telegram config
            api_key, api_secret = get_api_keys()
            telegram_token, telegram_chat_id = get_telegram_config()
            
            # Voeg API configuratie toe
            self.config['exchange']['api'].update({
                'api_key': api_key,
                'api_secret': api_secret
            })
            
            self.config['telegram'] = {
                'enabled': True,
                'token': telegram_token,
                'chat_id': telegram_chat_id,
                'max_retries': 3,
                'retry_delay': 5,
                'rate_limit': 30
            }
            
            logger.debug("Config geladen")
            self.log.info("Trading bot initialized")
            
            # Initialize components with error handling
            try:
                self.price_fetcher = PriceFetcher(self.config)
                self.log.info("PriceFetcher initialized successfully")
            except Exception as e:
                self.log.error(f"Failed to initialize PriceFetcher: {str(e)}")
                raise
                
            try:
                self.strategy = Strategy(self.config)
                self.log.info("Strategy initialized successfully")
            except Exception as e:
                self.log.error(f"Failed to initialize Strategy: {str(e)}")
                raise
                
            try:
                self.trade_executor = TradeExecutor(
                    config=TradeConfig(),
                    strategy=self.strategy,
                    price_fetcher=self.price_fetcher
                )
                self.log.info("TradeExecutor initialized successfully")
            except Exception as e:
                self.log.error(f"Failed to initialize TradeExecutor: {str(e)}")
                raise
            
            self.is_running = False
            self.active_trades = {}
            self.last_update = {}
            
            self.log.info("Trading bot initialized")
            
        except Exception as e:
            logger.error(f"Kritieke fout bij initialiseren bot: {e}")
            sys.exit(1)
            
    def _init_database(self):
        """Initialize database and create required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create performance_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        win_rate REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        sortino_ratio REAL,
                        max_drawdown REAL,
                        average_win REAL,
                        average_loss REAL,
                        total_profit REAL,
                        total_loss REAL,
                        metadata TEXT
                    )
                """)
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        type TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        quantity REAL,
                        pnl REAL,
                        status TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create market_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        timeframe TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        metadata TEXT
                    )
                """)
                
                conn.commit()
                self.log.info("Database initialized successfully")
                
        except Exception as e:
            self.log.error(f"Failed to initialize database: {str(e)}")
            raise

    async def initialize(self):
        """Initialize all components."""
        try:
            # Initialize async components
            await self.price_fetcher.initialize()
            
            self.is_running = True
            self.log.info("Trading bot started")
        except Exception as e:
            self.log.error(f"Error initializing trading bot: {str(e)}")
            raise

    async def close(self):
        """Close all components."""
        try:
            self.is_running = False
            self.log.info("Trading bot stopped")
        except Exception as e:
            self.log.error(f"Error closing trading bot: {str(e)}")

    async def _process_trading_cycle(self):
        """Process one trading cycle."""
        try:
            self.log.info("🔄 Starting new trading cycle...")
            
            for symbol in self.config['trading']['symbols'].keys():
                self.log.info(f"📊 Analyzing {symbol}...")
                
                # Get price data
                data = self.price_fetcher.get_klines(symbol, "15", 100)
                if data is None or data.empty:
                    self.log.warning(f"⚠️ No data available for {symbol}")
                    continue
                
                self.log.info(f"📈 Got {len(data)} candles for {symbol}")
                
                # Generate signal
                signal = self.strategy.generate_signal(data)
                if signal is None:
                    self.log.info(f"ℹ️ No trading signal for {symbol}")
                    continue
                
                self.log.info(f"🎯 Signal generated for {symbol}: {signal['type']} (strength: {signal.get('strength', 'N/A')})")
                
                # Execute trade
                if signal['type'] == 'long':
                    self.log.info(f"🟢 Executing LONG trade for {symbol}")
                    await self._execute_buy(symbol, signal)
                elif signal['type'] == 'short':
                    self.log.info(f"🔴 Executing SHORT trade for {symbol}")
                    await self._execute_sell(symbol, signal)
                
            # Log active trades
            if self.active_trades:
                self.log.info("📊 Active trades:")
                for symbol, trade in self.active_trades.items():
                    self.log.info(f"  - {symbol}: {trade}")
            else:
                self.log.info("📊 No active trades")
                
        except Exception as e:
            self.log.error(f"❌ Error in trading cycle: {str(e)}")
            self.log.error(traceback.format_exc())

    async def _execute_buy(self, symbol: str, signal: Dict):
        """Execute a buy order."""
        try:
            if symbol in self.active_trades:
                self.log.info(f"⚠️ Already have an active trade for {symbol}")
                return
            
            current_price = self.price_fetcher.get_price(symbol)
            if current_price is None:
                self.log.error(f"❌ Could not get current price for {symbol}")
                return
            
            position_size = self.config['risk']['max_position_size']
            
            self.log.info(f"🟢 Executing buy order for {symbol}:")
            self.log.info(f"  - Price: {current_price}")
            self.log.info(f"  - Size: {position_size}")
            self.log.info(f"  - Signal Strength: {signal.get('strength', 'N/A')}")
            
            trade = self.trade_executor.execute_trade(
                symbol=symbol,
                signal=signal['type'],
                quantity=position_size,
                price=current_price
            )
            
            if trade:
                self.active_trades[symbol] = trade
                self.log.info(f"✅ Buy order executed successfully for {symbol}")
                self._save_trade(trade)
            else:
                self.log.error(f"❌ Failed to execute buy order for {symbol}")
                
        except Exception as e:
            self.log.error(f"❌ Error executing buy order: {str(e)}")
            self.log.error(traceback.format_exc())

    async def _execute_sell(self, symbol: str, signal: Dict):
        """Execute a sell order."""
        try:
            if symbol not in self.active_trades:
                self.log.info(f"⚠️ No active trade found for {symbol}")
                return
            
            current_price = self.price_fetcher.get_price(symbol)
            if current_price is None:
                self.log.error(f"❌ Could not get current price for {symbol}")
                return
            
            trade = self.active_trades.pop(symbol)
            
            self.log.info(f"🔴 Executing sell order for {symbol}:")
            self.log.info(f"  - Entry Price: {trade.entry_price}")
            self.log.info(f"  - Current Price: {current_price}")
            self.log.info(f"  - PnL: {((current_price - trade.entry_price) / trade.entry_price * 100):.2f}%")
            
            success = self.trade_executor.close_trade(trade)
            
            if success:
                self.log.info(f"✅ Sell order executed successfully for {symbol}")
                self._update_trade(trade, current_price)
            else:
                self.log.error(f"❌ Failed to execute sell order for {symbol}")
                
        except Exception as e:
            self.log.error(f"❌ Error executing sell order: {str(e)}")
            self.log.error(traceback.format_exc())
            
    def _save_trade(self, trade: Trade):
        """Save trade to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (
                        symbol, type, entry_price, quantity, status, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trade.symbol,
                    trade.type,
                    trade.entry_price,
                    trade.quantity,
                    trade.status,
                    json.dumps(trade.metadata)
                ))
                conn.commit()
        except Exception as e:
            self.log.error(f"Failed to save trade: {str(e)}")
            
    def _update_trade(self, trade: Trade, exit_price: float):
        """Update trade in database with exit information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                pnl = (exit_price - trade.entry_price) * trade.quantity
                conn.execute("""
                    UPDATE trades 
                    SET exit_price = ?, pnl = ?, status = ?
                    WHERE symbol = ? AND status = 'open'
                """, (
                    exit_price,
                    pnl,
                    'closed',
                    trade.symbol
                ))
                conn.commit()
        except Exception as e:
            self.log.error(f"Failed to update trade: {str(e)}")

    async def run(self):
        """Run the trading bot."""
        try:
            await self.initialize()
            self.log.info("🚀 Trading bot is now running!")
            
            while self.is_running:
                try:
                    cycle_start_time = time.time()
                    self.log.info("🔄 Starting new trading cycle...")
                    
                    # Get current performance metrics
                    try:
                        metrics = self.performance_monitor.get_current_metrics()
                        self.log.info(f"📊 Current performance metrics:")
                        self.log.info(f"  - Win Rate: {metrics.get('win_rate', 0):.2%}")
                        self.log.info(f"  - Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                        self.log.info(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                        self.log.info(f"  - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                    except Exception as e:
                        self.log.error(f"Error getting performance metrics: {str(e)}")
                    
                    # Process each trading symbol
                    for symbol in self.config['trading']['symbols'].keys():
                        try:
                            self.log.info(f"📈 Analyzing {symbol}...")
                            
                            # Get price data
                            data = self.price_fetcher.get_klines(symbol, "15", 100)
                            if data is None or data.empty:
                                self.log.warning(f"⚠️ No data available for {symbol}")
                                continue
                            
                            # Save market data
                            try:
                                self._save_market_data(symbol, data)
                            except sqlite3.Error as e:
                                self.log.error(f"Database error saving market data: {str(e)}")
                            
                            self.log.info(f"📊 Got {len(data)} candles for {symbol}")
                            
                            # Generate signal
                            try:
                                signal = self.strategy.generate_signal(data)
                                if signal is None:
                                    self.log.info(f"ℹ️ No trading signal for {symbol}")
                                    continue
                                
                                self.log.info(f"🎯 Signal generated for {symbol}: {signal['type']} (strength: {signal.get('strength', 'N/A')})")
                                
                                # Execute trade
                                if signal['type'] == 'long':
                                    self.log.info(f"🟢 Executing LONG trade for {symbol}")
                                    await self._execute_buy(symbol, signal)
                                elif signal['type'] == 'short':
                                    self.log.info(f"🔴 Executing SHORT trade for {symbol}")
                                    await self._execute_sell(symbol, signal)
                            except Exception as e:
                                self.log.error(f"Error processing signal for {symbol}: {str(e)}")
                                continue
                            
                        except Exception as e:
                            self.log.error(f"Error processing symbol {symbol}: {str(e)}")
                            continue
                    
                    # Log active trades
                    if self.active_trades:
                        self.log.info("📊 Active trades:")
                        for symbol, trade in self.active_trades.items():
                            self.log.info(f"  - {symbol}: {trade}")
                    else:
                        self.log.info("📊 No active trades")
                    
                    # Update performance metrics
                    try:
                        self.performance_monitor.update_metrics()
                        updated_metrics = self.performance_monitor.get_current_metrics()
                        self.log.info(f"📈 Updated performance metrics:")
                        self.log.info(f"  - Win Rate: {updated_metrics.get('win_rate', 0):.2%}")
                        self.log.info(f"  - Profit Factor: {updated_metrics.get('profit_factor', 0):.2f}")
                        self.log.info(f"  - Sharpe Ratio: {updated_metrics.get('sharpe_ratio', 0):.2f}")
                        self.log.info(f"  - Max Drawdown: {updated_metrics.get('max_drawdown', 0):.2%}")
                    except Exception as e:
                        self.log.error(f"Error updating performance metrics: {str(e)}")
                    
                    # Calculate cycle duration
                    cycle_duration = time.time() - cycle_start_time
                    self.log.info(f"⏱️ Trading cycle completed in {cycle_duration:.2f} seconds")
                    
                    # Sleep until next cycle
                    sleep_time = max(0, self.config['trading']['cycle_interval'] - cycle_duration)
                    self.log.info(f"⏳ Waiting {sleep_time:.2f} seconds until next cycle...")
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    self.log.info("Trading cycle cancelled")
                    break
                except Exception as e:
                    self.log.error(f"❌ Error in trading cycle: {str(e)}")
                    self.log.error(traceback.format_exc())
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.log.error(f"❌ Error running trading bot: {str(e)}")
            self.log.error(traceback.format_exc())
        finally:
            await self.close()
            
    def _save_market_data(self, symbol: str, data: pd.DataFrame):
        """Save market data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for _, row in data.iterrows():
                    conn.execute("""
                        INSERT INTO market_data (
                            symbol, timeframe, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        "15m",
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['volume']
                    ))
                conn.commit()
        except Exception as e:
            self.log.error(f"Failed to save market data: {str(e)}")

class BotManager:
    def __init__(self):
        self.bot = None
        self.voice_controller = None
        self.running = False
        self.pid_file = Path('bot.pid')
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
    def start(self):
        """Start de bot met verbeterde error handling."""
        try:
            # Check if bot is already running
            if self._is_bot_running():
                self.logger.warning("Bot is already running")
                return
            
            # Schrijf PID naar bestand
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            # Registreer signal handlers
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)
            
            # Registreer cleanup handler
            atexit.register(self.cleanup)
            
            # Initialiseer bot
            self.bot = TradingBot()
            self.voice_controller = VoiceController(self.bot)
            
            # Start de bot
            self.logger.info("Starting trading bot...")
            self.running = True
            
            # Start voice controller
            self.voice_controller.start()
            
            # Start de main loop
            asyncio.run(self.bot.run())
            
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            self.cleanup()
            raise
            
    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}")
        self.running = False
        self.cleanup()
        
    def cleanup(self):
        """Cleanup resources bij shutdown."""
        try:
            self.running = False
            
            # Stop voice controller
            if hasattr(self, 'voice_controller') and self.voice_controller:
                try:
                    self.voice_controller.stop()
                    self.logger.info("Voice controller stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping voice controller: {str(e)}")
            
            # Stop bot
            if hasattr(self, 'bot') and self.bot:
                try:
                    self.bot.stop()
                    self.logger.info("Bot stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping bot: {str(e)}")
            
            # Verwijder PID file
            if self.pid_file.exists():
                try:
                    self.pid_file.unlink()
                    self.logger.info("PID file removed")
                except Exception as e:
                    self.logger.error(f"Error removing PID file: {str(e)}")
            
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    def _is_bot_running(self):
        """Check if bot is already running."""
        try:
            if self.pid_file.exists():
                try:
                    pid = int(self.pid_file.read_text().strip())
                    process = psutil.Process(pid)
                    if process.is_running() and process.name().lower().startswith('python'):
                        return True
                except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return False
        except Exception as e:
            self.logger.error(f"Error checking if bot is running: {str(e)}")
            return False

async def initialize_components(config):
    try:
        # Initialize database
        db_manager = DatabaseManager(config['database'])
        await db_manager.initialize()
        logging.info("Database initialized successfully")
        
        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(config['monitoring'])
        logging.info("Performance monitor initialized")
        
        # Initialize rate limiter
        rate_limiter = RateLimiter(
            requests_per_second=config['rate_limits']['requests_per_second'],
            requests_per_minute=config['rate_limits']['requests_per_minute'],
            requests_per_hour=config['rate_limits']['requests_per_hour']
        )
        logging.info(f"RateLimiter geïnitialiseerd met limieten: {rate_limiter.get_current_usage()}")
        
        # Initialize cache systems
        cache_manager = CacheManager(config['cache'])
        logging.info("Cache systemen geïnitialiseerd")
        
        # Initialize price fetcher
        price_fetcher = PriceFetcher(PriceFetcherConfig(**config['price_fetcher']))
        
        # Initialize trade executor
        trade_executor = TradeExecutor(config['trading'])
        
        return {
            'db_manager': db_manager,
            'perf_monitor': perf_monitor,
            'rate_limiter': rate_limiter,
            'cache_manager': cache_manager,
            'price_fetcher': price_fetcher,
            'trade_executor': trade_executor
        }
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}")
        logging.error(traceback.format_exc())
        raise

async def main():
    try:
        logging.info("[DEBUG] Initialiseren TradingBot...")
        
        # Load configuration
        config = load_config()  # Implement this function to load your config
        logging.info("[DEBUG] Config geladen")
        
        # Initialize components
        components = await initialize_components(config)
        
        # Start the main trading loop
        while True:
            try:
                # Your main trading logic here
                pass
            except Exception as e:
                logging.error(f"Error in main trading loop: {str(e)}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logging.error(f"Kritieke fout bij initialiseren bot: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    manager = BotManager()
    manager.start() 