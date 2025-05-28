import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from functools import lru_cache
import threading
from contextlib import contextmanager
from sqlalchemy import create_engine, text, event, MetaData, Table, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
import gc
from collections import deque
import psutil
import time
from typing import Set, Tuple

class DatabaseManager:
    """Database manager with optimized operations and error recovery."""
    
    def __init__(self, config: Dict):
        """Initialize database manager with optimized settings."""
        self.log = logging.getLogger(__name__)
        self.config = config
        
        # Initialize database with connection pooling
        self._init_database()
        
        # Initialize dynamic batch processing with adaptive sizing
        self._batch = {
            'market_data': deque(maxlen=self._calculate_optimal_batch_size()),
            'trades': deque(maxlen=self._calculate_optimal_batch_size()),
            'metrics': deque(maxlen=self._calculate_optimal_batch_size())
        }
        
        # Initialize cache with dynamic size and TTL
        self._cache = {
            'market_data': {},
            'trades': {},
            'metrics': {}
        }
        self._cache_ttl = {
            'market_data': 300,  # 5 minutes
            'trades': 600,       # 10 minutes
            'metrics': 3600      # 1 hour
        }
        self._cache_timestamps = {
            'market_data': {},
            'trades': {},
            'metrics': {}
        }
        
        # Initialize error tracking
        self._error_count = 0
        self._max_errors = 10
        self._error_window = deque(maxlen=1000)
        
        # Initialize transaction tracking
        self._active_transactions = set()
        self._transaction_timeouts = {}
        
        # Thread safety with fine-grained locking
        self._locks = {
            'batch': threading.RLock(),
            'cache': threading.RLock(),
            'transaction': threading.RLock(),
            'error': threading.RLock()
        }
        
        # Initialize garbage collection
        gc.set_threshold(700, 10, 5)
        
        # Start background tasks
        self._start_background_tasks()
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            available_memory = psutil.virtual_memory().available
            # Use 10% of available memory for batch processing
            memory_per_batch = available_memory * 0.1
            # Estimate memory per record (adjust based on your data)
            memory_per_record = 1024  # 1KB per record
            optimal_size = int(memory_per_batch / memory_per_record)
            # Ensure batch size is within reasonable bounds
            return min(max(100, optimal_size), 10000)
        except Exception as e:
            self.log.error(f"Error calculating optimal batch size: {str(e)}")
            return 1000  # Default batch size
            
    def _init_database(self):
        """Initialize database with optimized settings and error recovery."""
        try:
            # Calculate optimal pool size based on CPU cores and memory
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            pool_size = min(cpu_count * 2, int(memory_gb * 2), 20)
            
            # Create engine with optimized connection pooling
            self.engine = create_engine(
                f'sqlite:///{self.config["database_path"]}',
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=pool_size * 2,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                connect_args={
                    'timeout': 30,
                    'check_same_thread': False
                }
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
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA optimize")
                cursor.close()
            
            # Create optimized tables with proper indexes
            self._create_tables()
            
            # Prepare statements
            self._prepare_statements()
            
            self.log.info("Database initialized with optimized settings")
            
        except Exception as e:
            self.log.error(f"Failed to initialize database: {str(e)}")
            raise
            
    def _create_tables(self):
        """Create optimized tables with proper indexes and constraints."""
        try:
            metadata = MetaData()
            
            # Create market data table with optimized indexes
            market_data = Table('market_data', metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, nullable=False),
                Column('symbol', String(20), nullable=False),
                Column('timeframe', String(10), nullable=False),
                Column('open', Float, nullable=False),
                Column('high', Float, nullable=False),
                Column('low', Float, nullable=False),
                Column('close', Float, nullable=False),
                Column('volume', Float, nullable=False),
                Column('metadata', String),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Create trades table with optimized indexes
            trades = Table('trades', metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, nullable=False),
                Column('symbol', String(20), nullable=False),
                Column('type', String(10), nullable=False),
                Column('entry_price', Float, nullable=False),
                Column('exit_price', Float),
                Column('quantity', Float, nullable=False),
                Column('pnl', Float),
                Column('status', String(10), nullable=False),
                Column('metadata', String),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Create performance metrics table with optimized indexes
            performance_metrics = Table('performance_metrics', metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, nullable=False),
                Column('total_trades', Integer, nullable=False),
                Column('winning_trades', Integer, nullable=False),
                Column('losing_trades', Integer, nullable=False),
                Column('win_rate', Float, nullable=False),
                Column('profit_factor', Float, nullable=False),
                Column('sharpe_ratio', Float),
                Column('sortino_ratio', Float),
                Column('max_drawdown', Float),
                Column('average_win', Float),
                Column('average_loss', Float),
                Column('total_profit', Float),
                Column('total_loss', Float),
                Column('metadata', String),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Create indexes
            market_data.create(self.engine, checkfirst=True)
            trades.create(self.engine, checkfirst=True)
            performance_metrics.create(self.engine, checkfirst=True)
            
            # Create compound indexes for better query performance
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_market_symbol_timeframe_timestamp 
                    ON market_data(symbol, timeframe, timestamp)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol_status_timestamp 
                    ON trades(symbol, status, timestamp)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_perf_metrics_timestamp 
                    ON performance_metrics(timestamp)
                """))
                
        except Exception as e:
            self.log.error(f"Error creating tables: {str(e)}")
            raise
            
    def _start_background_tasks(self):
        """Start background tasks for maintenance and monitoring."""
        def cleanup_old_data():
            while True:
                try:
                    self._cleanup_old_data()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    self.log.error(f"Error in cleanup task: {str(e)}")
                    time.sleep(60)  # Wait before retry
                    
        def monitor_transactions():
            while True:
                try:
                    self._check_transaction_timeouts()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.log.error(f"Error in transaction monitoring: {str(e)}")
                    time.sleep(10)  # Wait before retry
                    
        # Start background threads
        threading.Thread(target=cleanup_old_data, daemon=True).start()
        threading.Thread(target=monitor_transactions, daemon=True).start()
        
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            retention_days = self.config.get('data_retention_days', 30)
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            with self.get_session() as session:
                # Delete old market data
                session.execute(text("""
                    DELETE FROM market_data 
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_time})
                
                # Delete old trades
                session.execute(text("""
                    DELETE FROM trades 
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_time})
                
                # Delete old performance metrics
                session.execute(text("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_time})
                
        except Exception as e:
            self.log.error(f"Error cleaning up old data: {str(e)}")
            
    def _check_transaction_timeouts(self):
        """Check for and handle transaction timeouts."""
        try:
            current_time = time.time()
            with self._locks['transaction']:
                # Find timed out transactions
                timed_out = [
                    tx_id for tx_id, timeout in self._transaction_timeouts.items()
                    if current_time > timeout
                ]
                
                # Rollback timed out transactions
                for tx_id in timed_out:
                    self._rollback_transaction(tx_id)
                    del self._transaction_timeouts[tx_id]
                    
        except Exception as e:
            self.log.error(f"Error checking transaction timeouts: {str(e)}")
            
    @contextmanager
    def transaction(self, timeout: int = 300):
        """Context manager for database transactions with timeout."""
        tx_id = id(threading.current_thread())
        try:
            with self._locks['transaction']:
                self._active_transactions.add(tx_id)
                self._transaction_timeouts[tx_id] = time.time() + timeout
                
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
                
        finally:
            with self._locks['transaction']:
                self._active_transactions.discard(tx_id)
                self._transaction_timeouts.pop(tx_id, None)
                
    def _rollback_transaction(self, tx_id: int):
        """Rollback a specific transaction."""
        try:
            with self._locks['transaction']:
                if tx_id in self._active_transactions:
                    session = self.Session()
                    try:
                        session.rollback()
                    finally:
                        session.close()
                    self._active_transactions.discard(tx_id)
                    
        except Exception as e:
            self.log.error(f"Error rolling back transaction {tx_id}: {str(e)}")
            
    def save_market_data(self, data: Union[Dict, pd.DataFrame]):
        """Save market data using batch processing with error recovery."""
        try:
            with self._locks['batch']:
                # Convert DataFrame to dict if needed
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict('records')
                elif isinstance(data, dict):
                    data = [data]
                    
                # Add data to batch
                self._batch['market_data'].extend(data)
                
                # Process batch if full
                if len(self._batch['market_data']) >= self._batch['market_data'].maxlen:
                    self._process_market_data_batch()
                    
        except Exception as e:
            self._handle_error('save_market_data', e)
            
    def _process_market_data_batch(self):
        """Process market data batch using optimized insert with retry."""
        try:
            if not self._batch['market_data']:
                return
                
            with self.transaction() as session:
                # Use executemany for better performance
                session.execute(self.insert_market_data, list(self._batch['market_data']))
                
            # Clear batch
            self._batch['market_data'].clear()
            
        except Exception as e:
            self._handle_error('process_market_data_batch', e)
            
    def _handle_error(self, operation: str, error: Exception):
        """Handle database errors with recovery mechanism."""
        try:
            with self._locks['error']:
                self._error_count += 1
                self._error_window.append({
                    'timestamp': datetime.utcnow(),
                    'operation': operation,
                    'error': str(error)
                })
                
                # Check if too many errors
                if self._error_count >= self._max_errors:
                    self.log.error("Too many errors, initiating recovery")
                    self._recover_from_errors()
                    
        except Exception as e:
            self.log.error(f"Error handling error: {str(e)}")
            
    def _recover_from_errors(self):
        """Recover from error state."""
        try:
            # Reset error count
            self._error_count = 0
            
            # Clear error window
            self._error_window.clear()
            
            # Rollback all active transactions
            with self._locks['transaction']:
                for tx_id in list(self._active_transactions):
                    self._rollback_transaction(tx_id)
                    
            # Clear batches
            with self._locks['batch']:
                for batch in self._batch.values():
                    batch.clear()
                    
            # Clear cache
            with self._locks['cache']:
                for cache in self._cache.values():
                    cache.clear()
                for timestamps in self._cache_timestamps.values():
                    timestamps.clear()
                    
            # Force garbage collection
            gc.collect()
            
            self.log.info("Recovery completed")
            
        except Exception as e:
            self.log.error(f"Error during recovery: {str(e)}")
            
    def cleanup(self):
        """Clean up resources with proper error handling."""
        try:
            # Process any remaining batches
            with self._locks['batch']:
                if self._batch['market_data']:
                    self._process_market_data_batch()
                if self._batch['trades']:
                    self._process_trade_batch()
                if self._batch['metrics']:
                    self._process_metrics_batch()
                    
            # Clear cache
            with self._locks['cache']:
                for cache in self._cache.values():
                    cache.clear()
                for timestamps in self._cache_timestamps.values():
                    timestamps.clear()
                    
            # Rollback any active transactions
            with self._locks['transaction']:
                for tx_id in list(self._active_transactions):
                    self._rollback_transaction(tx_id)
                    
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {str(e)}")
            
    def close(self):
        """Close database connections with proper cleanup."""
        try:
            self.cleanup()
            self.engine.dispose()
            self.log.info("Database connections closed")
            
        except Exception as e:
            self.log.error(f"Error closing database: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
        self.close()

    def save_trade(self, trade: Dict):
        """Save trade using batch processing."""
        try:
            with self._locks['batch']:
                # Add trade to batch
                self._batch['trades'].append(trade)
                
                # Process batch if full
                if len(self._batch['trades']) >= self._batch['trades'].maxlen:
                    self._process_trade_batch()
                    
        except Exception as e:
            self.log.error(f"Failed to save trade: {str(e)}")
            
    def _process_trade_batch(self):
        """Process trade batch using optimized insert."""
        try:
            if not self._batch['trades']:
                return
                
            with self.get_session() as session:
                # Use executemany for better performance
                session.execute(self.insert_trade, list(self._batch['trades']))
                
            # Clear batch
            self._batch['trades'].clear()
            
        except Exception as e:
            self.log.error(f"Failed to process trade batch: {str(e)}")
            
    def save_performance_metrics(self, metrics: Dict):
        """Save performance metrics using batch processing."""
        try:
            with self._locks['batch']:
                # Add metrics to batch
                self._batch['metrics'].append(metrics)
                
                # Process batch if full
                if len(self._batch['metrics']) >= self._batch['metrics'].maxlen:
                    self._process_metrics_batch()
                    
        except Exception as e:
            self.log.error(f"Failed to save performance metrics: {str(e)}")
            
    def _process_metrics_batch(self):
        """Process metrics batch using optimized insert."""
        try:
            if not self._batch['metrics']:
                return
                
            with self.get_session() as session:
                # Use executemany for better performance
                session.execute(self.insert_performance_metrics, list(self._batch['metrics']))
                
            # Clear batch
            self._batch['metrics'].clear()
            
        except Exception as e:
            self.log.error(f"Failed to process metrics batch: {str(e)}")
            
    def get_market_data(self, symbol: str, timeframe: str, start_time: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get market data with caching."""
        try:
            with self._locks['cache']:
                if symbol not in self._cache['market_data'] or timeframe not in self._cache['market_data'][symbol] or start_time not in self._cache['market_data'][symbol][timeframe]:
                    with self.get_session() as session:
                        result = session.execute(
                            self.get_market_data,
                            {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'start_time': start_time,
                                'limit': limit
                            }
                        ).fetchall()
                        
                        if not result:
                            return pd.DataFrame()
                        
                        df = pd.DataFrame(result)
                        df.set_index('timestamp', inplace=True)
                        self._cache['market_data'][symbol][timeframe] = df
                        self._cache_timestamps['market_data'][symbol][timeframe] = datetime.utcnow()
                    self.log.info(f"Market data for {symbol}, {timeframe} fetched and cached")
                else:
                    self.log.info(f"Market data for {symbol}, {timeframe} retrieved from cache")
                    df = self._cache['market_data'][symbol][timeframe]
                
            return df
                
        except Exception as e:
            self.log.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
            
    def update_trade(self, trade: Dict):
        """Update trade using optimized update."""
        try:
            with self.get_session() as session:
                session.execute(
                    self.update_trade,
                    {
                        'exit_price': trade['exit_price'],
                        'pnl': trade['pnl'],
                        'status': trade['status'],
                        'symbol': trade['symbol']
                    }
                )
                
        except Exception as e:
            self.log.error(f"Failed to update trade: {str(e)}")
            
    def save_market_data(self, market_data: pd.DataFrame) -> None:
        """Save market data to database."""
        try:
            with self.get_session() as session:
                # Convert DataFrame to list of tuples
                data = market_data.to_dict('records')
                
                # Insert data
                for row in data:
                    session.execute(self.insert_market_data, row)
                    
            self.log.info("Market data saved to database")
                
        except Exception as e:
            self.log.error(f"Error saving market data: {str(e)}")
            raise
            
    def save_performance_metrics(self, metrics: Dict) -> None:
        """Save performance metrics to database."""
        try:
            with self.get_session() as session:
                session.execute(self.insert_performance_metrics, metrics)
                
            self.log.info("Performance metrics saved to database")
                
        except Exception as e:
            self.log.error(f"Error saving performance metrics: {str(e)}")
            raise
            
    def save_system_metrics(self, metrics: Dict) -> None:
        """Save system metrics to database."""
        try:
            with self.get_session() as session:
                session.execute(self.insert_system_metrics, metrics)
                
            self.log.info("System metrics saved to database")
                
        except Exception as e:
            self.log.error(f"Error saving system metrics: {str(e)}")
            raise
            
    def save_alert(self, alert_data: Dict) -> None:
        """Save alert to database."""
        try:
            with self.get_session() as session:
                session.execute(self.insert_alert, alert_data)
                
            self.log.info("Alert saved to database")
                
        except Exception as e:
            self.log.error(f"Error saving alert: {str(e)}")
            raise
            
    def get_trades(self,
                  symbol: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  status: Optional[str] = None) -> pd.DataFrame:
        """Get trades from database."""
        try:
            with self.get_session() as session:
                # Build query
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                    
                if start_time:
                    query += " AND entry_time >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND entry_time <= ?"
                    params.append(end_time)
                    
                if status:
                    query += " AND status = ?"
                    params.append(status)
                    
                # Execute query
                df = pd.read_sql_query(query, session, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.log.error(f"Error getting trades: {str(e)}")
            raise
            
    def get_performance_metrics(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get performance metrics from database."""
        try:
            with self.get_session() as session:
                # Build query
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                # Execute query
                df = pd.read_sql_query(query, session, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.log.error(f"Error getting performance metrics: {str(e)}")
            raise
            
    def get_system_metrics(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get system metrics from database."""
        try:
            with self.get_session() as session:
                # Build query
                query = "SELECT * FROM system_metrics WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                # Execute query
                df = pd.read_sql_query(query, session, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.log.error(f"Error getting system metrics: {str(e)}")
            raise
            
    def get_alerts(self,
                  alert_type: Optional[str] = None,
                  level: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get alerts from database."""
        try:
            with self.get_session() as session:
                # Build query
                query = "SELECT * FROM alerts WHERE 1=1"
                params = []
                
                if alert_type:
                    query += " AND type = ?"
                    params.append(alert_type)
                    
                if level:
                    query += " AND level = ?"
                    params.append(level)
                    
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                # Execute query
                df = pd.read_sql_query(query, session, params=params)
                
                # Parse data
                if 'data' in df.columns:
                    df['data'] = df['data'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.log.error(f"Error getting alerts: {str(e)}")
            raise
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data from database."""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean up trades
                session.execute(text("""
                    DELETE FROM trades
                    WHERE entry_time < :cutoff_time
                """), {'cutoff_time': cutoff_date})
                
                # Clean up market data
                session.execute(text("""
                    DELETE FROM market_data
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_date})
                
                # Clean up performance metrics
                session.execute(text("""
                    DELETE FROM performance_metrics
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_date})
                
                # Clean up system metrics
                session.execute(text("""
                    DELETE FROM system_metrics
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_date})
                
                # Clean up alerts
                session.execute(text("""
                    DELETE FROM alerts
                    WHERE timestamp < :cutoff_time
                """), {'cutoff_time': cutoff_date})
                
            self.log.info("Old data cleaned up")
                
        except Exception as e:
            self.log.error(f"Error cleaning up old data: {str(e)}")
            raise
            
    def backup_database(self, backup_path: Optional[str] = None) -> None:
        """Backup database."""
        try:
            if backup_path is None:
                backup_path = f"{self.config['database_path']}.backup"
                
            # Create backup directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy database file
            with sqlite3.connect(self.config["database_path"]) as src_conn:
                with sqlite3.connect(backup_path) as dst_conn:
                    src_conn.backup(dst_conn)
                    
            self.log.info("Database backup completed")
            
        except Exception as e:
            self.log.error(f"Error backing up database: {str(e)}")
            raise
            
    def restore_database(self, backup_path: str) -> None:
        """Restore database from backup."""
        try:
            # Check if backup exists
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
                
            # Restore database
            with sqlite3.connect(backup_path) as src_conn:
                with sqlite3.connect(self.config["database_path"]) as dst_conn:
                    src_conn.backup(dst_conn)
                    
            self.log.info("Database restored from backup")
            
        except Exception as e:
            self.log.error(f"Error restoring database: {str(e)}")
            raise 