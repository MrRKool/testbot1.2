import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import sqlite3
import pandas as pd
import numpy as np
from functools import lru_cache
import threading
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
import gc
from collections import deque
import psutil
import os
from sqlalchemy.exc import SQLAlchemyError
from weakref import WeakValueDictionary

class PerformanceMonitor:
    """Performance monitor with real-time tracking and optimization."""
    
    def __init__(self, config: Dict):
        """Initialize performance monitor with advanced settings."""
        self.log = logging.getLogger(__name__)
        self.config = config
        
        # Initialize retention periods (in days)
        self._retention_periods = {
            'metrics': self.config.get('metrics_retention_days', 30),  # 30 days default
            'alerts': self.config.get('alerts_retention_days', 90),   # 90 days default
            'bottlenecks': self.config.get('bottlenecks_retention_days', 180)  # 180 days default
        }
        
        # Initialize cleanup intervals (in hours)
        self._cleanup_intervals = {
            'metrics': self.config.get('metrics_cleanup_interval', 24),    # Daily
            'alerts': self.config.get('alerts_cleanup_interval', 168),    # Weekly
            'bottlenecks': self.config.get('bottlenecks_cleanup_interval', 720)  # Monthly
        }
        
        # Track last cleanup times
        self._last_cleanup = {
            'metrics': time.time(),
            'alerts': time.time(),
            'bottlenecks': time.time()
        }
        
        # 1. Initialize core components
        self._init_core_components()
        
        # 2. Initialize monitoring systems
        self._init_monitoring_systems()
        
        # 3. Initialize database
        self._init_database()
        
        # 4. Start monitoring and cleanup
        self._start_monitoring()
        self._start_cleanup_tasks()
        
    def _init_core_components(self):
        """Initialize core components and settings."""
        # Initialize metrics tracking
        self._metrics = {
            'cpu': deque(maxlen=1000),
            'memory': deque(maxlen=1000),
            'disk': deque(maxlen=1000),
            'network': deque(maxlen=1000),
            'api': deque(maxlen=1000),
            'database': deque(maxlen=1000),
            'trades': deque(maxlen=1000),
            'returns': deque(maxlen=1000)
        }
        
        # Initialize performance thresholds
        self._thresholds = {
            'cpu': 80.0,      # 80% CPU usage
            'memory': 80.0,   # 80% memory usage
            'disk': 80.0,     # 80% disk usage
            'api_latency': 1.0,  # 1 second
            'db_latency': 0.5,   # 0.5 seconds
            'trade_latency': 0.1  # 0.1 seconds
        }
        
        # Initialize thread safety
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def _init_monitoring_systems(self):
        """Initialize monitoring systems."""
        # Initialize alerts system
        self._alerts = deque(maxlen=1000)
        self._alert_levels = {
            'info': 0,
            'warning': 1,
            'error': 2,
            'critical': 3
        }
        
        # Initialize bottleneck detection
        self._bottlenecks = {
            'cpu': 0,
            'memory': 0,
            'disk': 0,
            'network': 0,
            'api': 0,
            'database': 0
        }
        self._bottleneck_threshold = 5  # Number of consecutive violations
        
        # Initialize cache
        self._cache = WeakValueDictionary()
        self._cache_timestamps = {}
        self._cache_ttl = 300  # 5 minutes
        
    def _init_database(self):
        """Initialize database with optimized settings."""
        try:
            # Calculate optimal pool size
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            pool_size = min(cpu_count * 2, int(memory_gb * 2), 20)
            
            # Create engine with connection pooling
            self.engine = create_engine(
                f'sqlite:///{self.config["database_path"]}',
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=pool_size * 2,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True
            )
            
            # Create thread-safe session factory
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            # Configure SQLite for better performance
            self._configure_sqlite()
            
            # Create optimized tables
            self._create_tables()
            
        except Exception as e:
            self.log.error(f"Error initializing database: {str(e)}")
            raise
            
    def _configure_sqlite(self):
        """Configure SQLite for optimal performance."""
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
            
    def _create_tables(self):
        """Create optimized tables with proper indexes."""
        try:
            with self.engine.connect() as conn:
                # Create performance metrics table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        network_usage REAL,
                        api_latency REAL,
                        db_latency REAL,
                        trade_latency REAL,
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
                
                # Create alerts table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT,
                        message TEXT,
                        data TEXT
                    )
                """))
                
                # Create bottlenecks table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS bottlenecks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        type TEXT,
                        severity INTEGER,
                        message TEXT,
                        data TEXT
                    )
                """))
                
                # Create indexes
                self._create_indexes(conn)
                
        except Exception as e:
            self.log.error(f"Error creating tables: {str(e)}")
            raise
            
    def _create_indexes(self, conn):
        """Create optimized indexes for better query performance."""
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON performance_metrics(timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON alerts(timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_bottlenecks_timestamp 
                ON bottlenecks(timestamp)
            """))
            
        except Exception as e:
            self.log.error(f"Error creating indexes: {str(e)}")
            raise
            
    def _start_monitoring(self):
        """Start background performance monitoring."""
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    # 1. Update metrics
                    self._update_metrics()
                    
                    # 2. Detect bottlenecks
                    self._detect_bottlenecks()
                    
                    # 3. Check thresholds
                    self._check_thresholds()
                    
                    # 4. Save metrics
                    self._save_metrics()
                    
                    time.sleep(1)  # Update every second
                except Exception as e:
                    self.log.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(5)  # Wait before retry
                    
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def _update_metrics(self):
        """Update performance metrics."""
        try:
            # 1. CPU metrics
            self._update_cpu_metrics()
            
            # 2. Memory metrics
            self._update_memory_metrics()
            
            # 3. Disk metrics
            self._update_disk_metrics()
            
            # 4. Network metrics
            self._update_network_metrics()
            
        except Exception as e:
            self.log.error(f"Error updating metrics: {str(e)}")
            
    def _update_cpu_metrics(self):
        """Update CPU metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            self._metrics['cpu'].append({
                'timestamp': time.time(),
                'usage': cpu_percent,
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
            })
        except Exception as e:
            self.log.error(f"Error updating CPU metrics: {str(e)}")
            
    def _update_memory_metrics(self):
        """Update memory metrics."""
        try:
            memory = psutil.virtual_memory()
            self._metrics['memory'].append({
                'timestamp': time.time(),
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            })
        except Exception as e:
            self.log.error(f"Error updating memory metrics: {str(e)}")
            
    def _update_disk_metrics(self):
        """Update disk metrics."""
        try:
            disk = psutil.disk_usage('/')
            self._metrics['disk'].append({
                'timestamp': time.time(),
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            })
        except Exception as e:
            self.log.error(f"Error updating disk metrics: {str(e)}")
            
    def _update_network_metrics(self):
        """Update network metrics."""
        try:
            net_io = psutil.net_io_counters()
            self._metrics['network'].append({
                'timestamp': time.time(),
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            })
        except Exception as e:
            self.log.error(f"Error updating network metrics: {str(e)}")
            
    def _detect_bottlenecks(self):
        """Detect performance bottlenecks."""
        try:
            # 1. Check CPU bottleneck
            self._check_cpu_bottleneck()
            
            # 2. Check memory bottleneck
            self._check_memory_bottleneck()
            
            # 3. Check disk bottleneck
            self._check_disk_bottleneck()
            
        except Exception as e:
            self.log.error(f"Error detecting bottlenecks: {str(e)}")
            
    def _check_cpu_bottleneck(self):
        """Check for CPU bottleneck."""
        try:
            if self._metrics['cpu'] and self._metrics['cpu'][-1]['usage'] > self._thresholds['cpu']:
                self._bottlenecks['cpu'] += 1
                if self._bottlenecks['cpu'] >= self._bottleneck_threshold:
                    self._handle_bottleneck('cpu', 'High CPU usage detected')
            else:
                self._bottlenecks['cpu'] = 0
        except Exception as e:
            self.log.error(f"Error checking CPU bottleneck: {str(e)}")
            
    def _check_memory_bottleneck(self):
        """Check for memory bottleneck."""
        try:
            if self._metrics['memory'] and self._metrics['memory'][-1]['percent'] > self._thresholds['memory']:
                self._bottlenecks['memory'] += 1
                if self._bottlenecks['memory'] >= self._bottleneck_threshold:
                    self._handle_bottleneck('memory', 'High memory usage detected')
            else:
                self._bottlenecks['memory'] = 0
        except Exception as e:
            self.log.error(f"Error checking memory bottleneck: {str(e)}")
            
    def _check_disk_bottleneck(self):
        """Check for disk bottleneck."""
        try:
            if self._metrics['disk'] and self._metrics['disk'][-1]['percent'] > self._thresholds['disk']:
                self._bottlenecks['disk'] += 1
                if self._bottlenecks['disk'] >= self._bottleneck_threshold:
                    self._handle_bottleneck('disk', 'High disk usage detected')
            else:
                self._bottlenecks['disk'] = 0
        except Exception as e:
            self.log.error(f"Error checking disk bottleneck: {str(e)}")
            
    def _handle_bottleneck(self, type: str, message: str):
        """Handle detected bottleneck."""
        try:
            # 1. Create bottleneck record
            bottleneck = {
                'timestamp': datetime.utcnow(),
                'type': type,
                'severity': self._calculate_severity(type),
                'message': message,
                'data': json.dumps(self._get_bottleneck_data(type))
            }
            
            # 2. Save to database
            with self.Session() as session:
                session.execute(text("""
                    INSERT INTO bottlenecks (timestamp, type, severity, message, data)
                    VALUES (:timestamp, :type, :severity, :message, :data)
                """), bottleneck)
                session.commit()
                
            # 3. Create alert
            self._create_alert('warning', f"Performance bottleneck: {message}")
            
            # 4. Log bottleneck
            self.log.warning(f"Performance bottleneck detected: {message}")
            
        except Exception as e:
            self.log.error(f"Error handling bottleneck: {str(e)}")
            
    def _calculate_severity(self, type: str) -> int:
        """Calculate bottleneck severity."""
        try:
            # Base severity on current metrics
            if type == 'cpu':
                return int(self._metrics['cpu'][-1]['usage'] / 20)  # 1-5
            elif type == 'memory':
                return int(self._metrics['memory'][-1]['percent'] / 20)  # 1-5
            elif type == 'disk':
                return int(self._metrics['disk'][-1]['percent'] / 20)  # 1-5
            return 1
            
        except Exception as e:
            self.log.error(f"Error calculating severity: {str(e)}")
            return 1
            
    def _get_bottleneck_data(self, type: str) -> Dict:
        """Get detailed data for bottleneck."""
        try:
            if type == 'cpu':
                return {
                    'usage': self._metrics['cpu'][-1]['usage'],
                    'count': self._metrics['cpu'][-1]['count'],
                    'freq': self._metrics['cpu'][-1]['freq']
                }
            elif type == 'memory':
                return {
                    'total': self._metrics['memory'][-1]['total'],
                    'available': self._metrics['memory'][-1]['available'],
                    'percent': self._metrics['memory'][-1]['percent']
                }
            elif type == 'disk':
                return {
                    'total': self._metrics['disk'][-1]['total'],
                    'used': self._metrics['disk'][-1]['used'],
                    'percent': self._metrics['disk'][-1]['percent']
                }
            return {}
            
        except Exception as e:
            self.log.error(f"Error getting bottleneck data: {str(e)}")
            return {}
            
    def _check_thresholds(self):
        """Check performance thresholds."""
        try:
            # 1. Check CPU threshold
            self._check_cpu_threshold()
            
            # 2. Check memory threshold
            self._check_memory_threshold()
            
            # 3. Check disk threshold
            self._check_disk_threshold()
            
            # 4. Check latency thresholds
            self._check_latency_thresholds()
            
        except Exception as e:
            self.log.error(f"Error checking thresholds: {str(e)}")
            
    def _check_cpu_threshold(self):
        """Check CPU threshold."""
        try:
            if self._metrics['cpu'] and self._metrics['cpu'][-1]['usage'] > self._thresholds['cpu']:
                self._create_alert('warning', f"High CPU usage: {self._metrics['cpu'][-1]['usage']}%")
        except Exception as e:
            self.log.error(f"Error checking CPU threshold: {str(e)}")
            
    def _check_memory_threshold(self):
        """Check memory threshold."""
        try:
            if self._metrics['memory'] and self._metrics['memory'][-1]['percent'] > self._thresholds['memory']:
                self._create_alert('warning', f"High memory usage: {self._metrics['memory'][-1]['percent']}%")
        except Exception as e:
            self.log.error(f"Error checking memory threshold: {str(e)}")
            
    def _check_disk_threshold(self):
        """Check disk threshold."""
        try:
            if self._metrics['disk'] and self._metrics['disk'][-1]['percent'] > self._thresholds['disk']:
                self._create_alert('warning', f"High disk usage: {self._metrics['disk'][-1]['percent']}%")
        except Exception as e:
            self.log.error(f"Error checking disk threshold: {str(e)}")
            
    def _check_latency_thresholds(self):
        """Check latency thresholds."""
        try:
            # Check API latency
            if self._metrics['api'] and self._metrics['api'][-1]['latency'] > self._thresholds['api_latency']:
                self._create_alert('warning', f"High API latency: {self._metrics['api'][-1]['latency']}s")
                
            # Check database latency
            if self._metrics['database'] and self._metrics['database'][-1]['latency'] > self._thresholds['db_latency']:
                self._create_alert('warning', f"High database latency: {self._metrics['database'][-1]['latency']}s")
                
            # Check trade latency
            if self._metrics['trades'] and self._metrics['trades'][-1]['latency'] > self._thresholds['trade_latency']:
                self._create_alert('warning', f"High trade latency: {self._metrics['trades'][-1]['latency']}s")
                
        except Exception as e:
            self.log.error(f"Error checking latency thresholds: {str(e)}")
            
    def _create_alert(self, level: str, message: str, data: Dict = None):
        """Create performance alert."""
        try:
            # 1. Create alert record
            alert = {
                'timestamp': datetime.utcnow(),
                'level': level,
                'message': message,
                'data': json.dumps(data or {})
            }
            
            # 2. Add to alerts queue
            self._alerts.append(alert)
            
            # 3. Save to database
            with self.Session() as session:
                session.execute(text("""
                    INSERT INTO alerts (timestamp, level, message, data)
                    VALUES (:timestamp, :level, :message, :data)
                """), alert)
                session.commit()
                
            # 4. Log alert
            self.log.warning(f"Performance alert: {message}")
            
        except Exception as e:
            self.log.error(f"Error creating alert: {str(e)}")
            
    def _save_metrics(self):
        """Save performance metrics to database."""
        try:
            if not any(self._metrics.values()):
                return
                
            # 1. Create metrics record
            metrics = {
                'timestamp': datetime.utcnow(),
                'cpu_usage': self._metrics['cpu'][-1]['usage'] if self._metrics['cpu'] else None,
                'memory_usage': self._metrics['memory'][-1]['percent'] if self._metrics['memory'] else None,
                'disk_usage': self._metrics['disk'][-1]['percent'] if self._metrics['disk'] else None,
                'network_usage': self._metrics['network'][-1]['bytes_sent'] if self._metrics['network'] else None,
                'api_latency': self._metrics['api'][-1]['latency'] if self._metrics['api'] else None,
                'db_latency': self._metrics['database'][-1]['latency'] if self._metrics['database'] else None,
                'trade_latency': self._metrics['trades'][-1]['latency'] if self._metrics['trades'] else None,
                'metadata': json.dumps(self._get_metrics_metadata())
            }
            
            # 2. Save to database
            with self.Session() as session:
                session.execute(text("""
                    INSERT INTO performance_metrics (
                        timestamp, cpu_usage, memory_usage, disk_usage,
                        network_usage, api_latency, db_latency, trade_latency,
                        metadata
                    ) VALUES (
                        :timestamp, :cpu_usage, :memory_usage, :disk_usage,
                        :network_usage, :api_latency, :db_latency, :trade_latency,
                        :metadata
                    )
                """), metrics)
                session.commit()
                
        except Exception as e:
            self.log.error(f"Error saving metrics: {str(e)}")
            
    def _get_metrics_metadata(self) -> Dict:
        """Get additional metrics metadata."""
        try:
            return {
                'process': {
                    'pid': os.getpid(),
                    'create_time': psutil.Process().create_time(),
                    'num_threads': psutil.Process().num_threads(),
                    'num_handles': psutil.Process().num_handles() if hasattr(psutil.Process(), 'num_handles') else None
                },
                'system': {
                    'boot_time': psutil.boot_time(),
                    'users': len(psutil.users()),
                    'load_avg': psutil.getloadavg()
                }
            }
            
        except Exception as e:
            self.log.error(f"Error getting metrics metadata: {str(e)}")
            return {}
            
    def get_metrics(self, metric_type: str = None, start_time: datetime = None, end_time: datetime = None) -> Dict:
        """Get performance metrics."""
        try:
            with self.Session() as session:
                # 1. Build query
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = {}
                
                if metric_type:
                    query += " AND type = :type"
                    params['type'] = metric_type
                    
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params['start_time'] = start_time
                    
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params['end_time'] = end_time
                    
                query += " ORDER BY timestamp DESC"
                
                # 2. Execute query
                result = session.execute(text(query), params).fetchall()
                
                # 3. Convert to dict
                metrics = []
                for row in result:
                    metric = dict(row)
                    if 'metadata' in metric:
                        metric['metadata'] = json.loads(metric['metadata'])
                    metrics.append(metric)
                    
                return metrics
                
        except Exception as e:
            self.log.error(f"Error getting metrics: {str(e)}")
            return []
            
    def get_alerts(self, level: str = None, start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """Get performance alerts."""
        try:
            with self.Session() as session:
                # 1. Build query
                query = "SELECT * FROM alerts WHERE 1=1"
                params = {}
                
                if level:
                    query += " AND level = :level"
                    params['level'] = level
                    
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params['start_time'] = start_time
                    
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params['end_time'] = end_time
                    
                query += " ORDER BY timestamp DESC"
                
                # 2. Execute query
                result = session.execute(text(query), params).fetchall()
                
                # 3. Convert to dict
                alerts = []
                for row in result:
                    alert = dict(row)
                    if 'data' in alert:
                        alert['data'] = json.loads(alert['data'])
                    alerts.append(alert)
                    
                return alerts
                
        except Exception as e:
            self.log.error(f"Error getting alerts: {str(e)}")
            return []
            
    def get_bottlenecks(self, type: str = None, start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """Get performance bottlenecks."""
        try:
            with self.Session() as session:
                # 1. Build query
                query = "SELECT * FROM bottlenecks WHERE 1=1"
                params = {}
                
                if type:
                    query += " AND type = :type"
                    params['type'] = type
                    
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params['start_time'] = start_time
                    
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params['end_time'] = end_time
                    
                query += " ORDER BY timestamp DESC"
                
                # 2. Execute query
                result = session.execute(text(query), params).fetchall()
                
                # 3. Convert to dict
                bottlenecks = []
                for row in result:
                    bottleneck = dict(row)
                    if 'data' in bottleneck:
                        bottleneck['data'] = json.loads(bottleneck['data'])
                    bottlenecks.append(bottleneck)
                    
                return bottlenecks
                
        except Exception as e:
            self.log.error(f"Error getting bottlenecks: {str(e)}")
            return []
            
    def _start_cleanup_tasks(self):
        """Start background cleanup tasks."""
        def cleanup_loop():
            while not self._stop_event.is_set():
                try:
                    # Check and cleanup metrics
                    if self._should_cleanup('metrics'):
                        self._cleanup_old_metrics()
                        self._last_cleanup['metrics'] = time.time()
                        
                    # Check and cleanup alerts
                    if self._should_cleanup('alerts'):
                        self._cleanup_old_alerts()
                        self._last_cleanup['alerts'] = time.time()
                        
                    # Check and cleanup bottlenecks
                    if self._should_cleanup('bottlenecks'):
                        self._cleanup_old_bottlenecks()
                        self._last_cleanup['bottlenecks'] = time.time()
                        
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    self.log.error(f"Error in cleanup loop: {str(e)}")
                    time.sleep(300)  # Wait 5 minutes before retry
                    
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
    def _should_cleanup(self, data_type: str) -> bool:
        """Check if cleanup is needed for given data type."""
        try:
            current_time = time.time()
            last_cleanup = self._last_cleanup[data_type]
            interval = self._cleanup_intervals[data_type] * 3600  # Convert hours to seconds
            
            return (current_time - last_cleanup) >= interval
            
        except Exception as e:
            self.log.error(f"Error checking cleanup status: {str(e)}")
            return False
            
    def _cleanup_old_metrics(self):
        """Clean up old performance metrics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self._retention_periods['metrics'])
            
            with self.Session() as session:
                # Delete old metrics
                session.execute(text("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Get number of deleted rows
                deleted = session.execute(text("SELECT changes()")).scalar()
                
                session.commit()
                
            self.log.info(f"Cleaned up {deleted} old performance metrics")
            
        except Exception as e:
            self.log.error(f"Error cleaning up old metrics: {str(e)}")
            
    def _cleanup_old_alerts(self):
        """Clean up old alerts."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self._retention_periods['alerts'])
            
            with self.Session() as session:
                # Delete old alerts
                session.execute(text("""
                    DELETE FROM alerts 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Get number of deleted rows
                deleted = session.execute(text("SELECT changes()")).scalar()
                
                session.commit()
                
            self.log.info(f"Cleaned up {deleted} old alerts")
            
        except Exception as e:
            self.log.error(f"Error cleaning up old alerts: {str(e)}")
            
    def _cleanup_old_bottlenecks(self):
        """Clean up old bottlenecks."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self._retention_periods['bottlenecks'])
            
            with self.Session() as session:
                # Delete old bottlenecks
                session.execute(text("""
                    DELETE FROM bottlenecks 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Get number of deleted rows
                deleted = session.execute(text("SELECT changes()")).scalar()
                
                session.commit()
                
            self.log.info(f"Cleaned up {deleted} old bottlenecks")
            
        except Exception as e:
            self.log.error(f"Error cleaning up old bottlenecks: {str(e)}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # 1. Stop monitoring
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)
                
            # 2. Clear metrics
            for metric in self._metrics.values():
                metric.clear()
                
            # 3. Clear alerts
            self._alerts.clear()
            
            # 4. Clear bottlenecks
            for bottleneck in self._bottlenecks.values():
                bottleneck = 0
                
            # 5. Clear cache
            self._cache.clear()
            self._cache_timestamps.clear()
            
            # 6. Close database connection
            self.engine.dispose()
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 