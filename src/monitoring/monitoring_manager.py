import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
import psutil
import os
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import structlog
from structlog import get_logger
import pandas as pd
import numpy as np

class MonitoringManager:
    """Beheert monitoring en logging voor de trading bot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger()
        
        # Monitoring settings
        self.monitor_system = config.get('monitor_system', True)
        self.monitor_functions = config.get('monitor_functions', True)
        self.max_metrics_age_days = config.get('max_metrics_age_days', 7)
        self.max_metrics_count = config.get('max_metrics_count', 1000)
        self.metrics_export_path = config.get('metrics_export_path', 'metrics')
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize logging
        self._init_logging()
        
        # Initialize state
        self.metrics = {}
        self.is_running = False
        
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # System metrics
            self.cpu_usage = Gauge('cpu_usage', 'CPU usage percentage')
            self.memory_usage = Gauge('memory_usage', 'Memory usage percentage')
            self.disk_usage = Gauge('disk_usage', 'Disk usage percentage')
            self.network_io = Gauge('network_io', 'Network I/O bytes')
            
            # Trading metrics
            self.trade_count = Counter('trade_count', 'Number of trades')
            self.trade_volume = Counter('trade_volume', 'Total trade volume')
            self.profit_loss = Gauge('profit_loss', 'Current profit/loss')
            self.win_rate = Gauge('win_rate', 'Current win rate')
            
            # Performance metrics
            self.response_time = Histogram('response_time', 'API response time')
            self.error_count = Counter('error_count', 'Number of errors')
            self.task_duration = Summary('task_duration', 'Task execution duration')
            
            # Model metrics
            self.model_accuracy = Gauge('model_accuracy', 'Model accuracy')
            self.model_loss = Gauge('model_loss', 'Model loss')
            self.prediction_latency = Histogram('prediction_latency', 'Model prediction latency')
            
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {str(e)}")
            raise
            
    def _init_logging(self):
        """Initialize structured logging."""
        try:
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
                wrapper_class=structlog.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing logging: {str(e)}")
            raise
            
    async def start(self):
        """Start de monitoring manager."""
        try:
            self.is_running = True
            
            # Start metrics collection
            if self.monitor_system:
                asyncio.create_task(self._collect_system_metrics())
                
            # Start metrics cleanup
            asyncio.create_task(self._cleanup_old_metrics())
            
            self.logger.info("Monitoring manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring manager: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de monitoring manager."""
        try:
            self.is_running = False
            self.logger.info("Monitoring manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring manager: {str(e)}")
            return False
            
    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while self.is_running:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent())
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.percent)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.network_io.set(net_io.bytes_sent + net_io.bytes_recv)
                
                # Store metrics
                self._store_metrics({
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': memory.percent,
                    'disk_usage': disk.percent,
                    'network_io': net_io.bytes_sent + net_io.bytes_recv
                })
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {str(e)}")
                
            await asyncio.sleep(60)  # Collect every minute
            
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics periodically."""
        while self.is_running:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.max_metrics_age_days)
                
                # Remove old metrics
                self.metrics = {
                    k: v for k, v in self.metrics.items()
                    if datetime.fromisoformat(v['timestamp']) > cutoff_date
                }
                
                # Export metrics if needed
                if len(self.metrics) >= self.max_metrics_count:
                    self._export_metrics()
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up metrics: {str(e)}")
                
            await asyncio.sleep(3600)  # Cleanup every hour
            
    def _store_metrics(self, metrics: Dict):
        """Store metrics in memory."""
        try:
            timestamp = metrics['timestamp']
            self.metrics[timestamp] = metrics
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {str(e)}")
            raise
            
    def _export_metrics(self):
        """Export metrics to file."""
        try:
            # Create metrics directory if it doesn't exist
            os.makedirs(self.metrics_export_path, exist_ok=True)
            
            # Convert metrics to DataFrame
            df = pd.DataFrame.from_dict(self.metrics, orient='index')
            
            # Export to CSV
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(os.path.join(self.metrics_export_path, filename))
            
            # Clear metrics after export
            self.metrics.clear()
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
            raise
            
    def log_trade(self, trade_data: Dict):
        """Log trade data."""
        try:
            # Update trade metrics
            self.trade_count.inc()
            self.trade_volume.inc(trade_data.get('volume', 0))
            self.profit_loss.set(trade_data.get('profit_loss', 0))
            
            # Log trade
            self.logger.info(
                "trade_executed",
                trade_id=trade_data.get('trade_id'),
                symbol=trade_data.get('symbol'),
                side=trade_data.get('side'),
                price=trade_data.get('price'),
                volume=trade_data.get('volume'),
                profit_loss=trade_data.get('profit_loss')
            )
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
            raise
            
    def log_error(self, error_data: Dict):
        """Log error data."""
        try:
            # Update error metrics
            self.error_count.inc()
            
            # Log error
            self.logger.error(
                "error_occurred",
                error_type=error_data.get('type'),
                error_message=error_data.get('message'),
                stack_trace=error_data.get('stack_trace')
            )
            
        except Exception as e:
            self.logger.error(f"Error logging error: {str(e)}")
            raise
            
    def log_model_metrics(self, metrics: Dict):
        """Log model metrics."""
        try:
            # Update model metrics
            self.model_accuracy.set(metrics.get('accuracy', 0))
            self.model_loss.set(metrics.get('loss', 0))
            
            # Log metrics
            self.logger.info(
                "model_metrics_updated",
                model_type=metrics.get('model_type'),
                accuracy=metrics.get('accuracy'),
                loss=metrics.get('loss'),
                epoch=metrics.get('epoch')
            )
            
        except Exception as e:
            self.logger.error(f"Error logging model metrics: {str(e)}")
            raise
            
    def get_metrics_summary(self) -> Dict:
        """Get summary of current metrics."""
        try:
            return {
                'system': {
                    'cpu_usage': self.cpu_usage._value.get(),
                    'memory_usage': self.memory_usage._value.get(),
                    'disk_usage': self.disk_usage._value.get(),
                    'network_io': self.network_io._value.get()
                },
                'trading': {
                    'trade_count': self.trade_count._value.get(),
                    'trade_volume': self.trade_volume._value.get(),
                    'profit_loss': self.profit_loss._value.get(),
                    'win_rate': self.win_rate._value.get()
                },
                'performance': {
                    'error_count': self.error_count._value.get(),
                    'avg_response_time': self.response_time._sum.get() / max(self.response_time._count.get(), 1),
                    'avg_task_duration': self.task_duration._sum.get() / max(self.task_duration._count.get(), 1)
                },
                'model': {
                    'accuracy': self.model_accuracy._value.get(),
                    'loss': self.model_loss._value.get(),
                    'avg_prediction_latency': self.prediction_latency._sum.get() / max(self.prediction_latency._count.get(), 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {str(e)}")
            return {} 