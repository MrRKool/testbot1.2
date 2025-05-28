import os
import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Optional, Generator, List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import numpy as np

class DatabaseManager:
    """Database manager class met geoptimaliseerde database operaties."""
    
    def __init__(self, db_path: str = 'trading_bot.db'):
        """Initialiseer de database manager met geoptimaliseerde instellingen."""
        self.log = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database with connection pooling
        self._init_database()
        
        # Thread safety
        self._lock = threading.Lock()
        
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
                        metadata TEXT
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
                        metadata TEXT
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
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                """))
                
                # Create optimized indexes
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_market_symbol_timeframe_timestamp ON market_data(symbol, timeframe, timestamp)"))
                
        except Exception as e:
            self.log.error(f"Error creating tables: {str(e)}")
            raise
            
    def _prepare_statements(self):
        """Prepare SQL statements for better performance."""
        try:
            # Performance metrics statements
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
            
            # Market data statements
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
            
    def batch_insert(self, table: str, records: List[Dict]):
        """Insert multiple records using batch processing."""
        try:
            if not records:
                return
                
            with self.get_session() as session:
                if table == 'trades':
                    session.execute(self.insert_trade, records)
                elif table == 'market_data':
                    session.execute(self.insert_market_data, records)
                elif table == 'performance_metrics':
                    session.execute(self.insert_performance_metrics, records)
                    
        except Exception as e:
            self.log.error(f"Error batch inserting into {table}: {str(e)}")
            raise
            
    def bulk_update(self, table: str, records: List[Dict]):
        """Update multiple records using batch processing."""
        try:
            if not records:
                return
                
            with self.get_session() as session:
                if table == 'trades':
                    session.execute(self.update_trade, records)
                    
        except Exception as e:
            self.log.error(f"Error bulk updating {table}: {str(e)}")
            raise
            
    def get_market_data(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Get market data with optimized query."""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                result = session.execute(
                    self.get_market_data,
                    {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_time': start_time
                    }
                )
                data = pd.DataFrame(result.fetchall())
                
            if not data.empty:
                data.set_index('timestamp', inplace=True)
                
            return data
            
        except Exception as e:
            self.log.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
            
    def get_recent_trades(self, days: int = 30) -> pd.DataFrame:
        """Get recent trades with optimized query."""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                result = session.execute(
                    self.get_recent_trades,
                    {'start_time': start_time}
                )
                data = pd.DataFrame(result.fetchall())
                
            if not data.empty:
                data.set_index('timestamp', inplace=True)
                
            return data
            
        except Exception as e:
            self.log.error(f"Error getting recent trades: {str(e)}")
            return pd.DataFrame()
            
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data using batch processing."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                session.execute(
                    self.cleanup_old_data,
                    {'cutoff_time': cutoff_time}
                )
                
        except Exception as e:
            self.log.error(f"Error cleaning up old data: {str(e)}")
            
    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            self.log.info("Database connections closed")
            
        except Exception as e:
            self.log.error(f"Error closing database: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close() 