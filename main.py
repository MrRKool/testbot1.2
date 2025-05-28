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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

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
    """Hoofdclass voor de trading bot met geoptimaliseerde performance."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialiseer de trading bot met geoptimaliseerde instellingen."""
        self.log = logging.getLogger(__name__)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Initialize database with connection pooling
        self._init_database()
        
        # Initialize performance tracking with sliding window
        self._performance_metrics = {
            'cycle_times': [],
            'trades': [],
            'cache': {},
            'last_cleanup': datetime.now(),
            'batch_size': 1000  # Optimal batch size for processing
        }
        
        # Initialize trade batch processing
        self._trade_batch = []
        self._trade_batch_size = 100  # Optimal batch size for trades
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize garbage collection
        gc.set_threshold(700, 10, 5)
        
    def _init_database(self):
        """Initialize database with optimized settings."""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                f'sqlite:///{self.db_path}',
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True
            )
            
            # Create thread-safe session factory
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            # Configure SQLite for better performance
            @event.listens_for(self.engine, 'connect')
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=30000000000")
                cursor.execute("PRAGMA page_size=4096")
                cursor.execute("PRAGMA cache_size=-2000")
                cursor.execute("PRAGMA locking_mode=NORMAL")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()
            
            # Create optimized tables
            self._create_tables()
            
            # Prepare statements
            self._prepare_statements()
            
            self.log.info("Database initialized with optimized settings")
            
        except Exception as e:
            self.log.error(f"Failed to initialize database: {str(e)}")
            raise
            
    def _create_tables(self):
        """Create optimized tables with proper indexes."""
        try:
            with self.get_session() as session:
                # Create performance metrics table
                session.execute(text("""
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
                        metadata TEXT,
                        INDEX idx_perf_timestamp (timestamp)
                    )
                """))
                
                # Create trades table
                session.execute(text("""
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
                        metadata TEXT,
                        INDEX idx_trades_symbol_timestamp (symbol, timestamp),
                        INDEX idx_trades_status (status)
                    )
                """))
                
                # Create market data table
                session.execute(text("""
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
                        metadata TEXT,
                        UNIQUE(symbol, timeframe, timestamp),
                        INDEX idx_market_symbol_timeframe_timestamp (symbol, timeframe, timestamp)
                    )
                """))
                
        except Exception as e:
            self.log.error(f"Error creating tables: {str(e)}")
            raise
            
    def _prepare_statements(self):
        """Prepare SQL statements for better performance."""
        try:
            # Performance metrics statement
            self.insert_performance_metrics = text("""
                INSERT INTO performance_metrics (
                    timestamp, total_trades, winning_trades, losing_trades,
                    win_rate, profit_factor, sharpe_ratio, sortino_ratio,
                    max_drawdown, average_win, average_loss, total_profit,
                    total_loss, metadata
                ) VALUES (
                    :timestamp, :total_trades, :winning_trades, :losing_trades,
                    :win_rate, :profit_factor, :sharpe_ratio, :sortino_ratio,
                    :max_drawdown, :average_win, :average_loss, :total_profit,
                    :total_loss, :metadata
                )
            """)
            
            # Trade statements
            self.insert_trade = text("""
                INSERT INTO trades (
                    symbol, type, entry_price, quantity, status, metadata
                ) VALUES (
                    :symbol, :type, :entry_price, :quantity, :status, :metadata
                )
            """)
            
            self.update_trade = text("""
                UPDATE trades 
                SET exit_price = :exit_price, pnl = :pnl, status = :status
                WHERE symbol = :symbol AND status = 'open'
            """)
            
            # Market data statement
            self.insert_market_data = text("""
                INSERT INTO market_data (
                    timestamp, symbol, timeframe, open, high, low, close,
                    volume, metadata
                ) VALUES (
                    :timestamp, :symbol, :timeframe, :open, :high, :low, :close,
                    :volume, :metadata
                ) ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
                    open = :open, high = :high, low = :low, close = :close,
                    volume = :volume, metadata = :metadata
            """)
            
            # Query statements
            self.get_recent_trades = text("""
                SELECT * FROM trades 
                WHERE timestamp >= :start_time 
                ORDER BY timestamp DESC
            """)
            
            self.get_market_data = text("""
                SELECT * FROM market_data 
                WHERE symbol = :symbol 
                AND timeframe = :timeframe 
                AND timestamp >= :start_time 
                ORDER BY timestamp ASC
            """)
            
            # Cleanup statements
            self.cleanup_old_data = text("""
                DELETE FROM market_data 
                WHERE timestamp < :cutoff_time
            """)
                
        except Exception as e:
            self.log.error(f"Error preparing statements: {str(e)}")
            raise

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.log.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
            
    def _save_trade(self, trade: Trade):
        """Save trade using batch processing."""
        try:
            with self._lock:
                # Add trade to batch
                self._trade_batch.append({
                    'symbol': trade.symbol,
                    'type': trade.type,
                    'entry_price': trade.entry_price,
                    'quantity': trade.quantity,
                    'status': trade.status,
                    'metadata': json.dumps(trade.metadata)
                })
                
                # Process batch if full
                if len(self._trade_batch) >= self._trade_batch_size:
                    self._process_trade_batch()
                    
        except Exception as e:
            self.log.error(f"Failed to save trade: {str(e)}")
            
    def _process_trade_batch(self):
        """Process trade batch using optimized insert."""
        try:
            if not self._trade_batch:
                return
                
            with self.get_session() as session:
                session.execute(self.insert_trade, self._trade_batch)
                
            # Clear batch
            self._trade_batch = []
            
        except Exception as e:
            self.log.error(f"Failed to process trade batch: {str(e)}")
            
    def _update_trade(self, trade: Trade, exit_price: float):
        """Update trade using optimized update."""
        try:
            with self.get_session() as session:
                pnl = (exit_price - trade.entry_price) * trade.quantity
                session.execute(
                    self.update_trade,
                    {
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'status': 'closed',
                        'symbol': trade.symbol
                    }
                )
                
        except Exception as e:
            self.log.error(f"Failed to update trade: {str(e)}")
            
    @lru_cache(maxsize=1000)
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics with caching."""
        try:
            trades = self._performance_metrics['trades']
            if not trades:
                return {}
                
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            losing_trades = total_trades - winning_trades
            
            # Calculate advanced metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate risk metrics
            returns = np.array([t['pnl'] for t in trades])
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'average_win': total_profit / winning_trades if winning_trades > 0 else 0,
                'average_loss': total_loss / losing_trades if losing_trades > 0 else 0,
                'total_profit': total_profit,
                'total_loss': total_loss
            }
                
        except Exception as e:
            self.log.error(f"Error calculating performance metrics: {str(e)}")
            return {}
            
    def _save_performance_metrics(self, metrics: Dict):
        """Save performance metrics using batch processing."""
        try:
            with self.get_session() as session:
                session.execute(
                    self.insert_performance_metrics,
                    {
                        'timestamp': datetime.now(),
                        **metrics,
                        'metadata': json.dumps({})
                    }
                )
                
        except Exception as e:
            self.log.error(f"Error saving performance metrics: {str(e)}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            with self._lock:
                # Process any remaining trades
                if self._trade_batch:
                    self._process_trade_batch()
                    
                # Clear performance metrics cache
                self._calculate_performance_metrics.cache_clear()
                
                # Clear performance metrics
                self._performance_metrics['cycle_times'] = []
                self._performance_metrics['trades'] = []
                self._performance_metrics['cache'] = {}
                
                # Force garbage collection
                gc.collect()
                    
        except Exception as e:
            self.log.error(f"Error during cleanup: {str(e)}")
            
    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            self.log.info("Database connections closed")
            
        except Exception as e:
            self.log.error(f"Error closing database: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
        self.close()

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

def main():
    """Main function to run the trading bot."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Trading Bot")
        parser.add_argument("--mode", type=str, default="live", help="Trading mode (live/backtest)")
        parser.add_argument("--days", type=int, default=1825, help="Number of days for backtest (default: 5 years)")
        parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
        args = parser.parse_args()

        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return

        # Initialize bot
        bot = TradingBot()

        if args.mode == "backtest":
            # Run backtest
            logger.info(f"Starting backtest for {args.symbol} over {args.days} days")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            # Update config with date range
            config['backtest']['start_date'] = start_date.strftime('%Y-%m-%d')
            config['backtest']['end_date'] = end_date.strftime('%Y-%m-%d')
            
            # Initialize backtest engine with config
            engine = BacktestEngine(config)
            
            # Run backtest
            results = engine.run()
            
            if results:
                # Log results
                logger.info("\nBacktest Results:")
                logger.info(f"Total Trades: {results['total_trades']}")
                logger.info(f"Win Rate: {results['win_rate']:.2%}")
                logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
                logger.info(f"Total Return: {results['total_return']:.2%}")
                logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
                logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                logger.info(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
                
                # Save results
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                
                # Save trades to CSV
                trades_df = pd.DataFrame(engine.trades)
                trades_df.to_csv(results_dir / f"{args.symbol}_trades.csv", index=False)
                
                # Save equity curve plot
                plt.figure(figsize=(12, 6))
                equity_curve = pd.Series([t['equity'] for t in engine.equity_curve])
                equity_curve.plot()
                plt.title('Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Equity ($)')
                plt.grid(True)
                plt.savefig(results_dir / 'equity_curve.png')
                
                logger.info(f"Results saved to {results_dir}")
            else:
                logger.error("Backtest failed")
        else:
            # Run live trading
            asyncio.run(bot.run())
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 