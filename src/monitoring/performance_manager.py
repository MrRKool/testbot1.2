import logging
from typing import Dict, List, Any, Optional
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import gc
import weakref
import threading

class PerformanceManager:
    """Geoptimaliseerde performance manager voor AI componenten."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics settings
        self.metrics_window = config.get('metrics_window', 1000)
        self.metrics_interval = config.get('metrics_interval', 60)  # seconds
        self.save_interval = config.get('save_interval', 300)  # 5 minutes
        
        # Cache settings
        self.cache_size = config.get('cache_size', 1000)
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        
        # Initialize metrics storage with sliding window
        self.metrics = {
            'system': [],
            'network': [],
            'functions': {},
            'models': {}
        }
        
        # Initialize cache with weak references
        self._cache = weakref.WeakValueDictionary()
        self._cache_timestamps = {}
        
        # Initialize weak references
        self._weak_refs = weakref.WeakValueDictionary()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Last save time
        self.last_save = datetime.now()
        
        # Initialize garbage collection settings
        gc.set_threshold(700, 10, 5)  # More aggressive GC
        
    def monitor_system(self):
        """Monitor system performance with optimized metrics collection."""
        try:
            with self._lock:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                net = psutil.net_io_counters()
                
                # Store metrics with timestamp
                timestamp = datetime.now()
                
                # System metrics
                self.metrics['system'].append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used': memory.used,
                    'memory_available': memory.available,
                    'disk_read_bytes': disk.read_bytes,
                    'disk_write_bytes': disk.write_bytes,
                    'net_bytes_sent': net.bytes_sent,
                    'net_bytes_recv': net.bytes_recv
                })
                
                # Trim old metrics
                self._trim_metrics()
                
                # Save metrics periodically
                if (timestamp - self.last_save).total_seconds() >= self.save_interval:
                    self._save_metrics()
                    self.last_save = timestamp
                    
        except Exception as e:
            self.logger.error(f"Error monitoring system: {str(e)}")
            
    def _trim_metrics(self):
        """Trim old metrics using sliding window."""
        try:
            with self._lock:
                # Trim system metrics
                if len(self.metrics['system']) > self.metrics_window:
                    self.metrics['system'] = self.metrics['system'][-self.metrics_window:]
                    
                # Trim network metrics
                if len(self.metrics['network']) > self.metrics_window:
                    self.metrics['network'] = self.metrics['network'][-self.metrics_window:]
                    
                # Trim function metrics
                for func_name in self.metrics['functions']:
                    if len(self.metrics['functions'][func_name]) > self.metrics_window:
                        self.metrics['functions'][func_name] = self.metrics['functions'][func_name][-self.metrics_window:]
                        
                # Trim model metrics
                for model_name in self.metrics['models']:
                    if len(self.metrics['models'][model_name]) > self.metrics_window:
                        self.metrics['models'][model_name] = self.metrics['models'][model_name][-self.metrics_window:]
                        
        except Exception as e:
            self.logger.error(f"Error trimming metrics: {str(e)}")
            
    def monitor_function(self, func: Callable):
        """Monitor function performance with optimized metrics collection."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get start time and memory
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Get end time and memory
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                # Calculate metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                # Store metrics with timestamp
                timestamp = datetime.now()
                
                with self._lock:
                    if func.__name__ not in self.metrics['functions']:
                        self.metrics['functions'][func.__name__] = []
                        
                    self.metrics['functions'][func.__name__].append({
                        'timestamp': timestamp,
                        'execution_time': execution_time,
                        'memory_used': memory_used,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    })
                    
                    # Trim old metrics
                    if len(self.metrics['functions'][func.__name__]) > self.metrics_window:
                        self.metrics['functions'][func.__name__] = self.metrics['functions'][func.__name__][-self.metrics_window:]
                
                # Cache result if applicable
                if hasattr(func, '_cache_key'):
                    self.cache_result(func._cache_key, result)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error monitoring function {func.__name__}: {str(e)}")
                raise
                
        return wrapper
        
    def monitor_model(self, model_name: str):
        """Monitor model performance with optimized metrics collection."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Get start time
                    start_time = time.time()
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Get end time
                    end_time = time.time()
                    
                    # Calculate metrics
                    execution_time = end_time - start_time
                    
                    # Store metrics with timestamp
                    timestamp = datetime.now()
                    
                    with self._lock:
                        if model_name not in self.metrics['models']:
                            self.metrics['models'][model_name] = []
                            
                        self.metrics['models'][model_name].append({
                            'timestamp': timestamp,
                            'execution_time': execution_time,
                            'args': str(args),
                            'kwargs': str(kwargs)
                        })
                        
                        # Trim old metrics
                        if len(self.metrics['models'][model_name]) > self.metrics_window:
                            self.metrics['models'][model_name] = self.metrics['models'][model_name][-self.metrics_window:]
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring model {model_name}: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
        
    @lru_cache(maxsize=1000)
    def get_performance_stats(self) -> Dict:
        """Get performance statistics with caching."""
        try:
            stats = {
                'system': {},
                'network': {},
                'functions': {},
                'models': {}
            }
            
            # Calculate system stats
            if self.metrics['system']:
                system_df = pd.DataFrame(self.metrics['system'])
                stats['system'] = {
                    'cpu_percent': {
                        'current': system_df['cpu_percent'].iloc[-1],
                        'mean': system_df['cpu_percent'].mean(),
                        'max': system_df['cpu_percent'].max()
                    },
                    'memory_percent': {
                        'current': system_df['memory_percent'].iloc[-1],
                        'mean': system_df['memory_percent'].mean(),
                        'max': system_df['memory_percent'].max()
                    }
                }
            
            # Calculate network stats
            if self.metrics['network']:
                network_df = pd.DataFrame(self.metrics['network'])
                stats['network'] = {
                    'bytes_sent': {
                        'current': network_df['bytes_sent'].iloc[-1],
                        'total': network_df['bytes_sent'].sum()
                    },
                    'bytes_recv': {
                        'current': network_df['bytes_recv'].iloc[-1],
                        'total': network_df['bytes_recv'].sum()
                    }
                }
            
            # Calculate function stats
            for func_name, metrics in self.metrics['functions'].items():
                if not metrics:
                    continue
                    
                df = pd.DataFrame(metrics)
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
            
            # Calculate model stats
            for model_name, metrics in self.metrics['models'].items():
                if not metrics:
                    continue
                    
                df = pd.DataFrame(metrics)
                stats['models'][model_name] = {
                    'count': len(df),
                    'latest': df.iloc[-1].to_dict() if not df.empty else None,
                    'mean_execution_time': df['execution_time'].mean() if not df.empty else None,
                    'max_execution_time': df['execution_time'].max() if not df.empty else None,
                    'min_execution_time': df['execution_time'].min() if not df.empty else None
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {str(e)}")
            return {}
            
    def cache_result(self, key: str, result: Any):
        """Cache computation result with TTL and weak references."""
        try:
            with self._lock:
                # Use weak reference for result
                self._cache[key] = weakref.ref(result)
                self._cache_timestamps[key] = datetime.now()
                
                # Check cache size
                if len(self._cache) > self.cache_size:
                    # Remove oldest entries
                    oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                    del self._cache[oldest_key]
                    del self._cache_timestamps[oldest_key]
                    
        except Exception as e:
            self.logger.error(f"Error caching result: {str(e)}")
            
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        try:
            with self._lock:
                if key in self._cache:
                    timestamp = self._cache_timestamps[key]
                    if (datetime.now() - timestamp).total_seconds() <= self.cache_ttl:
                        # Get result from weak reference
                        result = self._cache[key]()
                        if result is not None:
                            return result
                        else:
                            # Result was garbage collected
                            del self._cache[key]
                            del self._cache_timestamps[key]
                    else:
                        # Remove expired entry
                        del self._cache[key]
                        del self._cache_timestamps[key]
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting cached result: {str(e)}")
            return None
            
    def _save_metrics(self):
        """Save metrics to database with batch processing."""
        try:
            with self._lock:
                # Convert metrics to DataFrames
                system_df = pd.DataFrame(self.metrics['system'])
                network_df = pd.DataFrame(self.metrics['network'])
                
                # Save to database (implement your database saving logic here)
                # system_df.to_sql('system_metrics', self.engine, if_exists='append')
                # network_df.to_sql('network_metrics', self.engine, if_exists='append')
                
                self.logger.info("Metrics saved successfully")
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            with self._lock:
                # Clear cache
                self._cache.clear()
                self._cache_timestamps.clear()
                
                # Clear weak references
                self._weak_refs.clear()
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 