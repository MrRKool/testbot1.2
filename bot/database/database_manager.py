import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class DatabaseManager:
    """Database manager for the trading bot."""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        quantity REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        pnl REAL,
                        status TEXT NOT NULL,
                        strategy TEXT,
                        metadata TEXT
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
                        metadata TEXT,
                        UNIQUE(symbol, timestamp)
                    )
                """)
                
                # Create performance_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        total_return REAL NOT NULL,
                        sharpe_ratio REAL,
                        sortino_ratio REAL,
                        max_drawdown REAL,
                        win_rate REAL,
                        profit_factor REAL,
                        avg_trade REAL,
                        avg_win REAL,
                        avg_loss REAL,
                        var_95 REAL,
                        expected_shortfall REAL,
                        metadata TEXT
                    )
                """)
                
                # Create system_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        api_latency REAL,
                        error_rate REAL,
                        metadata TEXT
                    )
                """)
                
                # Create alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        type TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
            
    def save_trade(self, trade_data: Dict) -> None:
        """Save trade to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, side, entry_price, exit_price, quantity,
                        entry_time, exit_time, pnl, status, strategy, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data.get('exit_price'),
                    trade_data['quantity'],
                    trade_data['entry_time'],
                    trade_data.get('exit_time'),
                    trade_data.get('pnl'),
                    trade_data['status'],
                    trade_data.get('strategy'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            raise
            
    def update_trade(self, trade_id: int, update_data: Dict) -> None:
        """Update trade in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
                query = f"UPDATE trades SET {set_clause} WHERE id = ?"
                
                # Execute update
                cursor.execute(query, list(update_data.values()) + [trade_id])
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            raise
            
    def save_market_data(self, market_data: pd.DataFrame) -> None:
        """Save market data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame to list of tuples
                data = market_data.to_dict('records')
                
                # Insert data
                for row in data:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data (
                            symbol, timestamp, open, high, low, close,
                            volume, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'],
                        row['timestamp'],
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['volume'],
                        json.dumps(row.get('metadata', {}))
                    ))
                    
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
            raise
            
    def save_performance_metrics(self, metrics: Dict) -> None:
        """Save performance metrics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_metrics (
                        timestamp, total_return, sharpe_ratio, sortino_ratio,
                        max_drawdown, win_rate, profit_factor, avg_trade,
                        avg_win, avg_loss, var_95, expected_shortfall, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics['timestamp'],
                    metrics['total_return'],
                    metrics.get('sharpe_ratio'),
                    metrics.get('sortino_ratio'),
                    metrics.get('max_drawdown'),
                    metrics.get('win_rate'),
                    metrics.get('profit_factor'),
                    metrics.get('avg_trade'),
                    metrics.get('avg_win'),
                    metrics.get('avg_loss'),
                    metrics.get('var_95'),
                    metrics.get('expected_shortfall'),
                    json.dumps(metrics.get('metadata', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            raise
            
    def save_system_metrics(self, metrics: Dict) -> None:
        """Save system metrics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_usage, memory_usage,
                        api_latency, error_rate, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metrics['timestamp'],
                    metrics.get('cpu_usage'),
                    metrics.get('memory_usage'),
                    metrics.get('api_latency'),
                    metrics.get('error_rate'),
                    json.dumps(metrics.get('metadata', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving system metrics: {e}")
            raise
            
    def save_alert(self, alert_data: Dict) -> None:
        """Save alert to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO alerts (
                        timestamp, type, level, message, data
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    alert_data['timestamp'],
                    alert_data['type'],
                    alert_data['level'],
                    alert_data['message'],
                    json.dumps(alert_data.get('data', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")
            raise
            
    def get_trades(self,
                  symbol: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  status: Optional[str] = None) -> pd.DataFrame:
        """Get trades from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            raise
            
    def get_market_data(self,
                       symbol: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get market data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query
                query = "SELECT * FROM market_data WHERE symbol = ?"
                params = [symbol]
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                # Execute query
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            raise
            
    def get_performance_metrics(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get performance metrics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            raise
            
    def get_system_metrics(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get system metrics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse metadata
                if 'metadata' in df.columns:
                    df['metadata'] = df['metadata'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            raise
            
    def get_alerts(self,
                  alert_type: Optional[str] = None,
                  level: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get alerts from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse data
                if 'data' in df.columns:
                    df['data'] = df['data'].apply(json.loads)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            raise
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean up trades
                cursor.execute("""
                    DELETE FROM trades
                    WHERE entry_time < ?
                """, (cutoff_date,))
                
                # Clean up market data
                cursor.execute("""
                    DELETE FROM market_data
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up performance metrics
                cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up system metrics
                cursor.execute("""
                    DELETE FROM system_metrics
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up alerts
                cursor.execute("""
                    DELETE FROM alerts
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
            
    def backup_database(self, backup_path: Optional[str] = None) -> None:
        """Backup database."""
        try:
            if backup_path is None:
                backup_path = f"{self.db_path}.backup"
                
            # Create backup directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy database file
            with sqlite3.connect(self.db_path) as src_conn:
                with sqlite3.connect(backup_path) as dst_conn:
                    src_conn.backup(dst_conn)
                    
        except Exception as e:
            self.logger.error(f"Error backing up database: {e}")
            raise
            
    def restore_database(self, backup_path: str) -> None:
        """Restore database from backup."""
        try:
            # Check if backup exists
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
                
            # Restore database
            with sqlite3.connect(backup_path) as src_conn:
                with sqlite3.connect(self.db_path) as dst_conn:
                    src_conn.backup(dst_conn)
                    
        except Exception as e:
            self.logger.error(f"Error restoring database: {e}")
            raise 