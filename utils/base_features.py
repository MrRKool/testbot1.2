from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import redis
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
import os

class BaseConfig:
    """Base configuration for features."""
    def __init__(self, **kwargs):
        self.num_workers = kwargs.get('num_workers', 4)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.use_redis = kwargs.get('use_redis', False)
        self.redis_host = kwargs.get('redis_host', 'localhost')
        self.redis_port = kwargs.get('redis_port', 6379)
        self.redis_db = kwargs.get('redis_db', 0)
        self.cache_ttl = kwargs.get('cache_ttl', 3600)
        self.cache_max_size = kwargs.get('cache_max_size', 1000)
        self.log_rotation = kwargs.get('log_rotation', '1 day')
        self.log_retention = kwargs.get('log_retention', '7 days')
        self.log_aggregation = kwargs.get('log_aggregation', True)
        self.enable_profiling = kwargs.get('enable_profiling', False)
        self.profile_interval = kwargs.get('profile_interval', 3600)
        self.resource_monitoring = kwargs.get('resource_monitoring', True)
        self.resource_check_interval = kwargs.get('resource_check_interval', 300)

class BaseFeatures:
    """Base class for trading features with optimizations."""
    def __init__(self, config: BaseConfig):
        self.config = config
        self._init_logging()
        self._init_caching()
        self._init_threading()
        self._init_profiling()
        
    def _init_logging(self):
        """Initialize logging with rotation and aggregation."""
        try:
            # Create logs directory
            os.makedirs('logs', exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.handlers.TimedRotatingFileHandler(
                        'logs/trading.log',
                        when=self.config.log_rotation.split()[1],
                        interval=int(self.config.log_rotation.split()[0]),
                        backupCount=int(self.config.log_retention.split()[0])
                    ),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger(self.__class__.__name__)
            
        except Exception as e:
            print(f"Error initializing logging: {e}")
            raise
            
    def _init_caching(self):
        """Initialize caching with Redis or in-memory."""
        try:
            if self.config.cache_enabled:
                if self.config.use_redis:
                    self.cache = redis.Redis(
                        host=self.config.redis_host,
                        port=self.config.redis_port,
                        db=self.config.redis_db
                    )
                else:
                    self.cache = {}
                    
        except Exception as e:
            self.logger.error(f"Error initializing caching: {e}")
            raise
            
    def _init_threading(self):
        """Initialize threading pool."""
        try:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.num_workers
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing threading: {e}")
            raise
            
    def _init_profiling(self):
        """Initialize profiling if enabled."""
        try:
            if self.config.enable_profiling:
                import cProfile
                self.profiler = cProfile.Profile()
                self.profiler.enable()
                
        except Exception as e:
            self.logger.error(f"Error initializing profiling: {e}")
            raise
            
    def cached_calculation(self, func):
        """Decorator for cached calculations."""
        @lru_cache(maxsize=self.config.cache_max_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
        
    async def async_operation(self, func, *args, **kwargs):
        """Run operation asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool,
                func,
                *args,
                **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Error in async operation: {e}")
            raise
            
    def safe_operation(self, func, *args, **kwargs):
        """Run operation with error handling."""
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in safe operation: {e}")
            raise
            
    def vectorized_operation(self, func, data: pd.DataFrame) -> pd.DataFrame:
        """Run vectorized operation on DataFrame."""
        try:
            return data.apply(func)
            
        except Exception as e:
            self.logger.error(f"Error in vectorized operation: {e}")
            raise
            
    def parallel_operation(self, func, items: List[Any]) -> List[Any]:
        """Run operation in parallel on list of items."""
        try:
            return list(self.thread_pool.map(func, items))
            
        except Exception as e:
            self.logger.error(f"Error in parallel operation: {e}")
            raise
            
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.config.cache_enabled:
                if self.config.use_redis:
                    value = self.cache.get(key)
                    return value.decode() if value else None
                else:
                    return self.cache.get(key)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            return None
            
    def set_in_cache(self, key: str, value: Any):
        """Set value in cache."""
        try:
            if self.config.cache_enabled:
                if self.config.use_redis:
                    self.cache.setex(key, self.config.cache_ttl, value)
                else:
                    self.cache[key] = value
                    
        except Exception as e:
            self.logger.error(f"Error setting in cache: {e}")
            
    def clear_cache(self):
        """Clear cache."""
        try:
            if self.config.cache_enabled:
                if self.config.use_redis:
                    self.cache.flushdb()
                else:
                    self.cache.clear()
                    
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics with aggregation if enabled."""
        try:
            if self.config.log_aggregation:
                # Aggregate metrics
                aggregated = self._aggregate_metrics(metrics)
                self.logger.info(f"Aggregated metrics: {aggregated}")
            else:
                self.logger.info(f"Metrics: {metrics}")
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            
    def _aggregate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics over time."""
        try:
            # Implement metric aggregation logic here
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {e}")
            return metrics
            
    def check_resources(self):
        """Check system resources if monitoring is enabled."""
        try:
            if self.config.resource_monitoring:
                import psutil
                
                metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                }
                
                self.log_metrics(metrics)
                
        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop thread pool
            self.thread_pool.shutdown()
            
            # Clear cache
            self.clear_cache()
            
            # Stop profiling
            if self.config.enable_profiling:
                self.profiler.disable()
                self.profiler.dump_stats('profile.stats')
                
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
            
    def __del__(self):
        """Destructor."""
        self.cleanup() 