import logging
import gc
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
import weakref
import threading
from functools import lru_cache
import pandas as pd
import numpy as np
from collections import deque
import time
from weakref import WeakValueDictionary
from contextlib import contextmanager

class MemoryManager:
    """Memory manager with advanced monitoring and optimization."""
    
    def __init__(self, config: Dict):
        """Initialize memory manager with advanced settings."""
        self.log = logging.getLogger(__name__)
        self.config = config
        
        # Initialize memory monitoring
        self._memory_pressure = 0.0  # 0.0 to 1.0
        self._pressure_history = deque(maxlen=1000)
        self._pressure_thresholds = {
            'low': 0.6,    # 60% memory usage
            'medium': 0.8, # 80% memory usage
            'high': 0.9    # 90% memory usage
        }
        
        # Initialize cache management
        self._cache = {
            'data': WeakValueDictionary(),
            'models': WeakValueDictionary(),
            'results': WeakValueDictionary()
        }
        self._cache_timestamps = {
            'data': {},
            'models': {},
            'results': {}
        }
        self._cache_ttl = {
            'data': 300,    # 5 minutes
            'models': 3600, # 1 hour
            'results': 1800 # 30 minutes
        }
        
        # Initialize memory tracking
        self._memory_usage = deque(maxlen=1000)
        self._memory_peaks = deque(maxlen=100)
        self._leak_suspicion = 0
        self._last_cleanup = time.time()
        
        # Initialize batch processing
        self._batch_sizes = {
            'data': self._calculate_initial_batch_size(),
            'models': 10,
            'results': 100
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Start monitoring
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Start background memory monitoring."""
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    self._update_memory_usage()
                    self._detect_memory_pressure()
                    self._check_for_leaks()
                    self._cleanup_if_needed()
                    time.sleep(1)  # Update every second
                except Exception as e:
                    self.log.error(f"Error in memory monitoring: {str(e)}")
                    time.sleep(5)  # Wait before retry
                    
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def _update_memory_usage(self):
        """Update memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Track memory usage
            self._memory_usage.append({
                'timestamp': time.time(),
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'shared': memory_info.shared,
                'text': memory_info.text,
                'lib': memory_info.lib,
                'data': memory_info.data,
                'dirty': memory_info.dirty
            })
            
            # Track memory peaks
            if len(self._memory_peaks) == 0 or memory_info.rss > self._memory_peaks[-1]['rss']:
                self._memory_peaks.append({
                    'timestamp': time.time(),
                    'rss': memory_info.rss
                })
                
        except Exception as e:
            self.log.error(f"Error updating memory usage: {str(e)}")
            
    def _detect_memory_pressure(self):
        """Detect memory pressure level."""
        try:
            # Calculate current memory pressure
            memory = psutil.virtual_memory()
            self._memory_pressure = memory.percent / 100.0
            
            # Update pressure history
            self._pressure_history.append({
                'timestamp': time.time(),
                'pressure': self._memory_pressure
            })
            
            # Handle pressure levels
            if self._memory_pressure >= self._pressure_thresholds['high']:
                self._handle_high_pressure()
            elif self._memory_pressure >= self._pressure_thresholds['medium']:
                self._handle_medium_pressure()
            elif self._memory_pressure >= self._pressure_thresholds['low']:
                self._handle_low_pressure()
                
        except Exception as e:
            self.log.error(f"Error detecting memory pressure: {str(e)}")
            
    def _handle_high_pressure(self):
        """Handle high memory pressure."""
        try:
            self.log.warning("High memory pressure detected")
            
            # Aggressive cleanup
            self._clear_all_caches()
            self._force_garbage_collection()
            self._reduce_batch_sizes()
            
            # Notify system
            self._notify_memory_pressure('high')
            
        except Exception as e:
            self.log.error(f"Error handling high pressure: {str(e)}")
            
    def _handle_medium_pressure(self):
        """Handle medium memory pressure."""
        try:
            self.log.info("Medium memory pressure detected")
            
            # Moderate cleanup
            self._clear_old_caches()
            self._suggest_garbage_collection()
            self._adjust_batch_sizes()
            
            # Notify system
            self._notify_memory_pressure('medium')
            
        except Exception as e:
            self.log.error(f"Error handling medium pressure: {str(e)}")
            
    def _handle_low_pressure(self):
        """Handle low memory pressure."""
        try:
            # Light cleanup
            self._clear_expired_caches()
            self._optimize_batch_sizes()
            
        except Exception as e:
            self.log.error(f"Error handling low pressure: {str(e)}")
            
    def _check_for_leaks(self):
        """Check for potential memory leaks."""
        try:
            if len(self._memory_usage) < 2:
                return
                
            # Calculate memory growth rate
            recent_usage = list(self._memory_usage)[-10:]
            if len(recent_usage) < 2:
                return
                
            growth_rate = (recent_usage[-1]['rss'] - recent_usage[0]['rss']) / len(recent_usage)
            
            # Check for suspicious growth
            if growth_rate > 1024 * 1024:  # More than 1MB per second
                self._leak_suspicion += 1
                if self._leak_suspicion >= 5:
                    self._handle_suspected_leak()
            else:
                self._leak_suspicion = max(0, self._leak_suspicion - 1)
                
        except Exception as e:
            self.log.error(f"Error checking for leaks: {str(e)}")
            
    def _handle_suspected_leak(self):
        """Handle suspected memory leak."""
        try:
            self.log.warning("Potential memory leak detected")
            
            # Take memory snapshot
            self._take_memory_snapshot()
            
            # Force cleanup
            self._clear_all_caches()
            self._force_garbage_collection()
            
            # Reset suspicion
            self._leak_suspicion = 0
            
        except Exception as e:
            self.log.error(f"Error handling suspected leak: {str(e)}")
            
    def _take_memory_snapshot(self):
        """Take memory snapshot for analysis."""
        try:
            process = psutil.Process()
            memory_maps = process.memory_maps()
            
            # Save snapshot
            snapshot = {
                'timestamp': time.time(),
                'memory_maps': memory_maps,
                'memory_info': process.memory_info()._asdict(),
                'open_files': process.open_files(),
                'connections': process.connections()
            }
            
            # Log snapshot
            self.log.debug(f"Memory snapshot: {snapshot}")
            
        except Exception as e:
            self.log.error(f"Error taking memory snapshot: {str(e)}")
            
    def _clear_all_caches(self):
        """Clear all caches."""
        try:
            with self._lock:
                for cache in self._cache.values():
                    cache.clear()
                for timestamps in self._cache_timestamps.values():
                    timestamps.clear()
                    
        except Exception as e:
            self.log.error(f"Error clearing caches: {str(e)}")
            
    def _clear_old_caches(self):
        """Clear old cache entries."""
        try:
            with self._lock:
                current_time = time.time()
                for cache_type, ttl in self._cache_ttl.items():
                    expired = [
                        key for key, timestamp in self._cache_timestamps[cache_type].items()
                        if current_time - timestamp > ttl
                    ]
                    for key in expired:
                        self._cache[cache_type].pop(key, None)
                        self._cache_timestamps[cache_type].pop(key, None)
                        
        except Exception as e:
            self.log.error(f"Error clearing old caches: {str(e)}")
            
    def _clear_expired_caches(self):
        """Clear expired cache entries."""
        try:
            with self._lock:
                current_time = time.time()
                for cache_type, ttl in self._cache_ttl.items():
                    expired = [
                        key for key, timestamp in self._cache_timestamps[cache_type].items()
                        if current_time - timestamp > ttl
                    ]
                    for key in expired:
                        self._cache[cache_type].pop(key, None)
                        self._cache_timestamps[cache_type].pop(key, None)
                        
        except Exception as e:
            self.log.error(f"Error clearing expired caches: {str(e)}")
            
    def _force_garbage_collection(self):
        """Force garbage collection."""
        try:
            # Collect all generations
            collected = gc.collect(generation=2)
            self.log.debug(f"Garbage collection: {collected} objects collected")
            
        except Exception as e:
            self.log.error(f"Error in garbage collection: {str(e)}")
            
    def _suggest_garbage_collection(self):
        """Suggest garbage collection if needed."""
        try:
            if gc.get_count()[0] > 700:  # Threshold for collection
                gc.collect()
                
        except Exception as e:
            self.log.error(f"Error suggesting garbage collection: {str(e)}")
            
    def _reduce_batch_sizes(self):
        """Reduce batch sizes under memory pressure."""
        try:
            with self._lock:
                for key in self._batch_sizes:
                    self._batch_sizes[key] = max(1, self._batch_sizes[key] // 2)
                    
        except Exception as e:
            self.log.error(f"Error reducing batch sizes: {str(e)}")
            
    def _adjust_batch_sizes(self):
        """Adjust batch sizes based on memory pressure."""
        try:
            with self._lock:
                for key in self._batch_sizes:
                    self._batch_sizes[key] = max(1, int(self._batch_sizes[key] * 0.8))
                    
        except Exception as e:
            self.log.error(f"Error adjusting batch sizes: {str(e)}")
            
    def _optimize_batch_sizes(self):
        """Optimize batch sizes based on available memory."""
        try:
            with self._lock:
                available_memory = psutil.virtual_memory().available
                for key in self._batch_sizes:
                    self._batch_sizes[key] = self._calculate_optimal_batch_size(key, available_memory)
                    
        except Exception as e:
            self.log.error(f"Error optimizing batch sizes: {str(e)}")
            
    def _calculate_optimal_batch_size(self, key: str, available_memory: int) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            # Base sizes for different types
            base_sizes = {
                'data': 1000,
                'models': 10,
                'results': 100
            }
            
            # Calculate based on available memory
            memory_factor = available_memory / (1024 * 1024 * 1024)  # Convert to GB
            return max(1, int(base_sizes[key] * memory_factor))
            
        except Exception as e:
            self.log.error(f"Error calculating batch size: {str(e)}")
            return 1
            
    def _calculate_initial_batch_size(self) -> int:
        """Calculate initial batch size based on system resources."""
        try:
            # Get system information
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Calculate based on available resources
            return max(1, int(min(cpu_count * 100, memory_gb * 100)))
            
        except Exception as e:
            self.log.error(f"Error calculating initial batch size: {str(e)}")
            return 100
            
    def _notify_memory_pressure(self, level: str):
        """Notify system about memory pressure."""
        try:
            # Log pressure level
            self.log.warning(f"Memory pressure level: {level}")
            
            # Update metrics
            self._update_pressure_metrics(level)
            
        except Exception as e:
            self.log.error(f"Error notifying memory pressure: {str(e)}")
            
    def _update_pressure_metrics(self, level: str):
        """Update memory pressure metrics."""
        try:
            metrics = {
                'timestamp': time.time(),
                'pressure_level': level,
                'memory_usage': psutil.virtual_memory().percent,
                'available_memory': psutil.virtual_memory().available,
                'swap_usage': psutil.swap_memory().percent
            }
            
            # Log metrics
            self.log.debug(f"Memory pressure metrics: {metrics}")
            
        except Exception as e:
            self.log.error(f"Error updating pressure metrics: {str(e)}")
            
    def get_batch_size(self, key: str) -> int:
        """Get current batch size for given key."""
        try:
            with self._lock:
                return self._batch_sizes.get(key, 1)
                
        except Exception as e:
            self.log.error(f"Error getting batch size: {str(e)}")
            return 1
            
    def get_memory_pressure(self) -> float:
        """Get current memory pressure level."""
        try:
            return self._memory_pressure
            
        except Exception as e:
            self.log.error(f"Error getting memory pressure: {str(e)}")
            return 0.0
            
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'shared': memory_info.shared,
                'text': memory_info.text,
                'lib': memory_info.lib,
                'data': memory_info.data,
                'dirty': memory_info.dirty,
                'percent': psutil.virtual_memory().percent
            }
            
        except Exception as e:
            self.log.error(f"Error getting memory usage: {str(e)}")
            return {}
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop monitoring
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
                
            # Clear caches
            self._clear_all_caches()
            
            # Force garbage collection
            self._force_garbage_collection()
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 