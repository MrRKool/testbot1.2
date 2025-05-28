import logging
from typing import Dict, List, Any, Optional
import os
from datetime import datetime, timedelta
import json
import time
import psutil
import numpy as np
import pandas as pd
from functools import wraps

class PerformanceManager:
    """Manages performance monitoring for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance settings
        self.performance_dir = config.get('performance_dir', 'performance')
        self.max_metrics_age = timedelta(days=config.get('max_metrics_age_days', 30))
        self.max_metrics = config.get('max_metrics', 10000)
        
        # Initialize metrics
        self.metrics = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': [],
            'functions': {}
        }
        self._load_metrics()
        
    def _load_metrics(self):
        """Load existing metrics."""
        try:
            # Create performance directory if it doesn't exist
            os.makedirs(self.performance_dir, exist_ok=True)
            
            # Load metrics file
            metrics_file = os.path.join(self.performance_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading metrics: {str(e)}")
            
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            # Create performance directory if it doesn't exist
            os.makedirs(self.performance_dir, exist_ok=True)
            
            # Save metrics
            metrics_file = os.path.join(self.performance_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def monitor_system(self):
        """Monitor system performance."""
        try:
            # Get current time
            timestamp = datetime.now().isoformat()
            
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            self.metrics['cpu'].append({
                'timestamp': timestamp,
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else None
            })
            
            # Get memory metrics
            memory = psutil.virtual_memory()
            self.metrics['memory'].append({
                'timestamp': timestamp,
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            })
            
            # Get disk metrics
            disk = psutil.disk_usage('/')
            self.metrics['disk'].append({
                'timestamp': timestamp,
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            })
            
            # Get network metrics
            network = psutil.net_io_counters()
            self.metrics['network'].append({
                'timestamp': timestamp,
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            })
            
            # Save metrics
            self._save_metrics()
            
            # Cleanup if needed
            self.cleanup_old_metrics()
            
        except Exception as e:
            self.logger.error(f"Error monitoring system: {str(e)}")
            
    def monitor_function(self, func):
        """Decorator to monitor function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get start time
                start_time = time.time()
                
                # Get start memory
                start_memory = psutil.Process().memory_info().rss
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Get end time and memory
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                # Calculate metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                # Store metrics
                if func.__name__ not in self.metrics['functions']:
                    self.metrics['functions'][func.__name__] = []
                    
                self.metrics['functions'][func.__name__].append({
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': execution_time,
                    'memory_used': memory_used,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                
                # Save metrics
                self._save_metrics()
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error monitoring function {func.__name__}: {str(e)}")
                raise
                
        return wrapper
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        try:
            stats = {
                'system': {},
                'functions': {}
            }
            
            # Calculate system stats
            for metric_type in ['cpu', 'memory', 'disk', 'network']:
                if not self.metrics[metric_type]:
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(self.metrics[metric_type])
                
                # Calculate stats
                stats['system'][metric_type] = {
                    'count': len(df),
                    'latest': df.iloc[-1].to_dict() if not df.empty else None,
                    'mean': df.mean().to_dict() if not df.empty else None,
                    'max': df.max().to_dict() if not df.empty else None,
                    'min': df.min().to_dict() if not df.empty else None
                }
                
            # Calculate function stats
            for func_name, metrics in self.metrics['functions'].items():
                if not metrics:
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(metrics)
                
                # Calculate stats
                stats['functions'][func_name] = {
                    'count': len(df),
                    'latest': df.iloc[-1].to_dict() if not df.empty else None,
                    'mean_execution_time': df['execution_time'].mean() if not df.empty else None,
                    'max_execution_time': df['execution_time'].max() if not df.empty else None,
                    'min_execution_time': df['execution_time'].min() if not df.empty else None,
                    'mean_memory_used': df['memory_used'].mean() if not df.empty else None,
                    'max_memory_used': df['memory_used'].max() if not df.empty else None,
                    'min_memory_used': df['memory_used'].min() if not df.empty else None
                }
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {str(e)}")
            return {}
            
    def cleanup_old_metrics(self):
        """Remove old metrics."""
        try:
            # Get current time
            now = datetime.now()
            
            # Cleanup system metrics
            for metric_type in ['cpu', 'memory', 'disk', 'network']:
                self.metrics[metric_type] = [
                    m for m in self.metrics[metric_type]
                    if datetime.fromisoformat(m['timestamp']) > now - self.max_metrics_age
                ]
                
                # Limit total metrics
                if len(self.metrics[metric_type]) > self.max_metrics:
                    self.metrics[metric_type] = self.metrics[metric_type][-self.max_metrics:]
                    
            # Cleanup function metrics
            for func_name in self.metrics['functions']:
                self.metrics['functions'][func_name] = [
                    m for m in self.metrics['functions'][func_name]
                    if datetime.fromisoformat(m['timestamp']) > now - self.max_metrics_age
                ]
                
                # Limit total metrics
                if len(self.metrics['functions'][func_name]) > self.max_metrics:
                    self.metrics['functions'][func_name] = self.metrics['functions'][func_name][-self.max_metrics:]
                    
            # Save metrics
            self._save_metrics()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {str(e)}")
            
    def export_metrics(self, filepath: str, metric_type: str = None, func_name: str = None, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Export metrics to file."""
        try:
            # Get metrics to export
            metrics_to_export = {}
            
            # Add system metrics
            if metric_type:
                if metric_type in self.metrics:
                    metrics_to_export[metric_type] = self.metrics[metric_type]
            else:
                for mt in ['cpu', 'memory', 'disk', 'network']:
                    metrics_to_export[mt] = self.metrics[mt]
                    
            # Add function metrics
            if func_name:
                if func_name in self.metrics['functions']:
                    metrics_to_export['functions'] = {func_name: self.metrics['functions'][func_name]}
            else:
                metrics_to_export['functions'] = self.metrics['functions']
                
            # Apply date filters
            if start_date or end_date:
                for key in metrics_to_export:
                    if key == 'functions':
                        for func in metrics_to_export[key]:
                            metrics_to_export[key][func] = [
                                m for m in metrics_to_export[key][func]
                                if (not start_date or datetime.fromisoformat(m['timestamp']) >= start_date) and
                                (not end_date or datetime.fromisoformat(m['timestamp']) <= end_date)
                            ]
                    else:
                        metrics_to_export[key] = [
                            m for m in metrics_to_export[key]
                            if (not start_date or datetime.fromisoformat(m['timestamp']) >= start_date) and
                            (not end_date or datetime.fromisoformat(m['timestamp']) <= end_date)
                        ]
                        
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(metrics_to_export, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
            return False 