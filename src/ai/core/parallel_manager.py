import logging
import gc
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Generator, Tuple
import weakref
import threading
from functools import lru_cache, partial
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError, CancelledError
import queue
import multiprocessing
from queue import Queue, Empty, PriorityQueue
import time
from collections import deque
import asyncio
from dataclasses import dataclass
from enum import Enum
import statistics
from threading import Lock, RLock, Event, Condition
import warnings
from contextlib import contextmanager
import signal
import sys
import traceback
from typing import Set

class TaskPriority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class Task:
    priority: TaskPriority
    func: Callable
    data: Any
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    task_id: str = None
    dependencies: Set[str] = None  # Added for task dependencies
    timeout: float = 300.0  # Default timeout in seconds

    def __post_init__(self):
        if self.task_id is None:
            self.task_id = f"task_{int(time.time() * 1000)}_{id(self)}"
        if self.dependencies is None:
            self.dependencies = set()

class ResourceMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'io_usage': deque(maxlen=1000),
            'network_usage': deque(maxlen=1000),
            'disk_usage': deque(maxlen=1000)  # Added disk usage monitoring
        }
        self._lock = RLock()
        self._running = False
        self._monitor_thread = None
        self._stop_event = Event()
        self._condition = Condition()  # Added for thread synchronization
        self._error_count = 0  # Added for error tracking
        self._max_errors = 10  # Maximum consecutive errors before stopping
        
    def start(self):
        """Start resource monitoring."""
        if not self._running:
            self._running = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
    def stop(self):
        """Stop resource monitoring."""
        if self._running:
            self._running = False
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                if self._monitor_thread.is_alive():
                    logging.warning("Monitor thread did not stop gracefully")
                    # Force thread termination if necessary
                    if hasattr(threading, '_threads_queues'):
                        for thread_id, queue in threading._threads_queues.items():
                            if thread_id == self._monitor_thread.ident:
                                queue.put(None)
            
    def _monitor_loop(self):
        """Monitor loop for collecting resource metrics."""
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    self._metrics['cpu_usage'].append(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self._metrics['memory_usage'].append(memory.percent)
                    
                    # IO usage
                    io_counters = psutil.disk_io_counters()
                    self._metrics['io_usage'].append(io_counters.read_bytes + io_counters.write_bytes)
                    
                    # Network usage
                    net_io = psutil.net_io_counters()
                    self._metrics['network_usage'].append(net_io.bytes_sent + net_io.bytes_recv)
                    
                    # Disk usage
                    disk_usage = psutil.disk_usage('/')
                    self._metrics['disk_usage'].append(disk_usage.percent)
                    
                    # Reset error count on successful update
                    self._error_count = 0
                    
                # Use condition wait instead of event wait for better synchronization
                with self._condition:
                    self._condition.wait(self.update_interval)
                    
            except Exception as e:
                self._error_count += 1
                logging.error(f"Error in resource monitoring: {str(e)}\n{traceback.format_exc()}")
                
                # Stop monitoring if too many consecutive errors
                if self._error_count >= self._max_errors:
                    logging.error("Too many consecutive errors in resource monitoring, stopping")
                    self.stop()
                    break
                    
                # Add small delay to prevent tight loop on error
                time.sleep(0.1)
                
    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        with self._lock:
            try:
                return {
                    'cpu_usage': statistics.mean(self._metrics['cpu_usage']) if self._metrics['cpu_usage'] else 0,
                    'memory_usage': statistics.mean(self._metrics['memory_usage']) if self._metrics['memory_usage'] else 0,
                    'io_usage': statistics.mean(self._metrics['io_usage']) if self._metrics['io_usage'] else 0,
                    'network_usage': statistics.mean(self._metrics['network_usage']) if self._metrics['network_usage'] else 0,
                    'disk_usage': statistics.mean(self._metrics['disk_usage']) if self._metrics['disk_usage'] else 0
                }
            except Exception as e:
                logging.error(f"Error getting metrics: {str(e)}\n{traceback.format_exc()}")
                return {
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'io_usage': 0,
                    'network_usage': 0,
                    'disk_usage': 0
                }

class ParallelManager:
    """Parallel processing manager with optimized task distribution."""
    
    def __init__(self, config: Dict):
        """Initialize parallel manager with optimized settings."""
        self.log = logging.getLogger(__name__)
        self.config = config
        
        # Initialize resource monitor
        self._resource_monitor = ResourceMonitor()
        self._resource_monitor.start()
        
        # Initialize thread and process pools with dynamic sizing
        self._init_pools()
        
        # Initialize priority queue for tasks
        self._task_queue = PriorityQueue()
        
        # Initialize workload tracking with enhanced metrics
        self._workload = {
            'thread_pool': {
                'active_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'avg_completion_time': 0.0,
                'history': deque(maxlen=1000),
                'performance_metrics': {
                    'throughput': deque(maxlen=100),
                    'latency': deque(maxlen=100),
                    'error_rate': deque(maxlen=100),
                    'memory_usage': deque(maxlen=100),  # Added memory usage tracking
                    'cpu_usage': deque(maxlen=100)      # Added CPU usage tracking
                }
            },
            'process_pool': {
                'active_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'avg_completion_time': 0.0,
                'history': deque(maxlen=1000),
                'performance_metrics': {
                    'throughput': deque(maxlen=100),
                    'latency': deque(maxlen=100),
                    'error_rate': deque(maxlen=100),
                    'memory_usage': deque(maxlen=100),  # Added memory usage tracking
                    'cpu_usage': deque(maxlen=100)      # Added CPU usage tracking
                }
            }
        }
        
        # Initialize multi-level cache with size limits and TTL
        self._cache = {
            'l1': weakref.WeakValueDictionary(),  # Fast, small cache
            'l2': weakref.WeakValueDictionary(),  # Medium cache
            'l3': weakref.WeakValueDictionary()   # Large, slow cache
        }
        self._cache_sizes = {
            'l1': 1000,  # Maximum items in L1 cache
            'l2': 5000,  # Maximum items in L2 cache
            'l3': 10000  # Maximum items in L3 cache
        }
        self._cache_ttl = {
            'l1': 60,    # TTL in seconds for L1 cache
            'l2': 300,   # TTL in seconds for L2 cache
            'l3': 3600   # TTL in seconds for L3 cache
        }
        self._cache_timestamps = {
            'l1': weakref.WeakKeyDictionary(),
            'l2': weakref.WeakKeyDictionary(),
            'l3': weakref.WeakKeyDictionary()
        }
        
        # Initialize chunk size tracking with adaptive sizing
        self._chunk_sizes = {
            'thread': self._calculate_optimal_chunk_size('thread'),
            'process': self._calculate_optimal_chunk_size('process')
        }
        
        # Initialize performance tracking
        self._performance = {
            'start_time': time.time(),
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'memory_peak': 0.0,  # Added memory peak tracking
            'cpu_peak': 0.0      # Added CPU peak tracking
        }
        
        # Thread safety with fine-grained locking
        self._locks = {
            'workload': RLock(),
            'cache': RLock(),
            'chunk_sizes': RLock(),
            'performance': RLock()
        }
        
        # Initialize garbage collection with optimized settings
        gc.set_threshold(700, 10, 5)
        
        # Start workload monitoring
        self._start_workload_monitoring()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        # Initialize task dependency tracking
        self._task_dependencies = {}
        self._completed_tasks = set()
        
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.log.info(f"Received signal {signum}, initiating graceful shutdown")
            self.cleanup()
            sys.exit(0)
            
        # Register handlers for common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            try:
                signal.signal(sig, signal_handler)
            except (ValueError, OSError) as e:
                self.log.warning(f"Could not register handler for signal {sig}: {e}")
                
    def _init_pools(self):
        """Initialize thread and process pools with optimal settings."""
        try:
            # Get system resources
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Calculate optimal pool sizes based on system resources and current load
            metrics = self._resource_monitor.get_metrics()
            cpu_usage = metrics['cpu_usage']
            memory_usage = metrics['memory_usage']
            
            # Adjust pool sizes based on current system load
            load_factor = 1.0 - (cpu_usage / 100.0)  # Higher load = fewer workers
            
            # Calculate thread pool size
            thread_pool_size = min(
                int(cpu_count * 4 * load_factor),  # 4 threads per CPU core, adjusted for load
                int(memory_gb * 2 * load_factor),  # 2 threads per GB of memory
                100  # Maximum thread pool size
            )
            
            # Calculate process pool size
            process_pool_size = min(
                int(cpu_count * load_factor),  # 1 process per CPU core
                int(memory_gb * load_factor),  # 1 process per GB of memory
                20  # Maximum process pool size
            )
            
            # Initialize thread pool with optimized settings
            self.thread_pool = ThreadPoolExecutor(
                max_workers=thread_pool_size,
                thread_name_prefix='trading_thread',
                initializer=self._thread_initializer
            )
            
            # Initialize process pool with optimized settings
            self.process_pool = ProcessPoolExecutor(
                max_workers=process_pool_size,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=self._process_initializer
            )
            
            self.log.info(f"Initialized thread pool with {thread_pool_size} workers")
            self.log.info(f"Initialized process pool with {process_pool_size} workers")
            
        except Exception as e:
            self.log.error(f"Error initializing pools: {str(e)}\n{traceback.format_exc()}")
            raise
            
    def _thread_initializer(self):
        """Initialize thread with optimized settings."""
        try:
            # Set thread name
            threading.current_thread().name = f"trading_thread_{threading.get_ident()}"
            
            # Set thread priority if possible
            if hasattr(threading, 'set_thread_priority'):
                threading.set_thread_priority(threading.get_ident(), 0)
                
            # Initialize thread-local storage
            threading.current_thread().local = threading.local()
            
            # Set up exception handling
            sys.excepthook = self._thread_exception_handler
            
            # Set up thread-specific logging
            logging.getLogger().addHandler(logging.StreamHandler())
            
        except Exception as e:
            self.log.error(f"Error initializing thread: {str(e)}\n{traceback.format_exc()}")
            
    def _thread_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions in threads."""
        self.log.error("Uncaught exception in thread", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Update performance metrics
        with self._locks['performance']:
            self._performance['failed_tasks'] += 1
            
    def _process_initializer(self):
        """Initialize process with optimized settings."""
        try:
            # Set process name
            if hasattr(multiprocessing, 'set_process_name'):
                multiprocessing.set_process_name(f"trading_process_{os.getpid()}")
                
            # Set process priority if possible
            if hasattr(os, 'nice'):
                os.nice(0)
                
            # Initialize process-local storage
            multiprocessing.current_process().local = {}
            
            # Set up exception handling
            sys.excepthook = self._process_exception_handler
            
            # Set up process-specific logging
            logging.getLogger().addHandler(logging.StreamHandler())
            
        except Exception as e:
            self.log.error(f"Error initializing process: {str(e)}\n{traceback.format_exc()}")
            
    def _process_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions in processes."""
        self.log.error("Uncaught exception in process", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Update performance metrics
        with self._locks['performance']:
            self._performance['failed_tasks'] += 1
            
    def _calculate_optimal_chunk_size(self, pool_type: str) -> int:
        """Calculate optimal chunk size based on system resources and current load."""
        try:
            # Get system resources
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Get current system load
            metrics = self._resource_monitor.get_metrics()
            cpu_usage = metrics['cpu_usage']
            memory_usage = metrics['memory_usage']
            
            # Calculate load factor
            load_factor = 1.0 - ((cpu_usage + memory_usage) / 200.0)  # Higher load = smaller chunks
            
            # Calculate base chunk size
            if pool_type == 'thread':
                # Thread pool can handle smaller chunks
                base_size = min(100, int(memory_gb * 10 * load_factor))
            else:
                # Process pool needs larger chunks due to overhead
                base_size = min(1000, int(memory_gb * 100 * load_factor))
                
            # Adjust based on CPU cores and load
            optimal_size = int(base_size * cpu_count * load_factor)
            
            # Ensure size is within reasonable bounds
            return min(max(10, optimal_size), 10000)
            
        except Exception as e:
            self.log.error(f"Error calculating optimal chunk size: {str(e)}\n{traceback.format_exc()}")
            return 1000  # Default chunk size
            
    def _start_workload_monitoring(self):
        """Start workload monitoring in background."""
        def monitor_workload():
            while True:
                try:
                    self._update_workload_stats()
                    self._adjust_resources()
                    time.sleep(1)  # Update every second
                except Exception as e:
                    self.log.error(f"Error in workload monitoring: {str(e)}\n{traceback.format_exc()}")
                    time.sleep(0.1)  # Prevent tight loop on error
                    
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_workload, daemon=True)
        monitor_thread.start()
        
    def _update_workload_stats(self):
        """Update workload statistics with enhanced metrics."""
        try:
            with self._locks['workload']:
                # Get current timestamp
                current_time = time.time()
                
                # Get current resource usage
                metrics = self._resource_monitor.get_metrics()
                
                # Update thread pool stats
                thread_active = len(self.thread_pool._work_queue)
                self._workload['thread_pool']['active_tasks'] = thread_active
                
                # Calculate thread pool metrics
                thread_history = list(self._workload['thread_pool']['history'])
                if thread_history:
                    # Calculate throughput
                    completed_since_last = self._workload['thread_pool']['completed_tasks'] - thread_history[-1].get('completed_tasks', 0)
                    time_since_last = current_time - thread_history[-1].get('timestamp', current_time)
                    throughput = completed_since_last / time_since_last if time_since_last > 0 else 0
                    
                    # Calculate latency
                    latency = statistics.mean([
                        h['avg_completion_time']
                        for h in thread_history[-10:]  # Last 10 measurements
                    ]) if thread_history else 0
                    
                    # Calculate error rate
                    total_tasks = self._workload['thread_pool']['completed_tasks'] + self._workload['thread_pool']['failed_tasks']
                    error_rate = self._workload['thread_pool']['failed_tasks'] / total_tasks if total_tasks > 0 else 0
                    
                    # Update performance metrics
                    self._workload['thread_pool']['performance_metrics']['throughput'].append(throughput)
                    self._workload['thread_pool']['performance_metrics']['latency'].append(latency)
                    self._workload['thread_pool']['performance_metrics']['error_rate'].append(error_rate)
                    self._workload['thread_pool']['performance_metrics']['memory_usage'].append(metrics['memory_usage'])
                    self._workload['thread_pool']['performance_metrics']['cpu_usage'].append(metrics['cpu_usage'])
                
                # Update thread pool history
                self._workload['thread_pool']['history'].append({
                    'timestamp': current_time,
                    'active_tasks': thread_active,
                    'completed_tasks': self._workload['thread_pool']['completed_tasks'],
                    'failed_tasks': self._workload['thread_pool']['failed_tasks'],
                    'avg_completion_time': self._workload['thread_pool']['avg_completion_time']
                })
                
                # Similar updates for process pool
                process_active = len(self.process_pool._work_queue)
                self._workload['process_pool']['active_tasks'] = process_active
                
                process_history = list(self._workload['process_pool']['history'])
                if process_history:
                    # Calculate process pool metrics
                    completed_since_last = self._workload['process_pool']['completed_tasks'] - process_history[-1].get('completed_tasks', 0)
                    time_since_last = current_time - process_history[-1].get('timestamp', current_time)
                    throughput = completed_since_last / time_since_last if time_since_last > 0 else 0
                    
                    latency = statistics.mean([
                        h['avg_completion_time']
                        for h in process_history[-10:]
                    ]) if process_history else 0
                    
                    total_tasks = self._workload['process_pool']['completed_tasks'] + self._workload['process_pool']['failed_tasks']
                    error_rate = self._workload['process_pool']['failed_tasks'] / total_tasks if total_tasks > 0 else 0
                    
                    self._workload['process_pool']['performance_metrics']['throughput'].append(throughput)
                    self._workload['process_pool']['performance_metrics']['latency'].append(latency)
                    self._workload['process_pool']['performance_metrics']['error_rate'].append(error_rate)
                    self._workload['process_pool']['performance_metrics']['memory_usage'].append(metrics['memory_usage'])
                    self._workload['process_pool']['performance_metrics']['cpu_usage'].append(metrics['cpu_usage'])
                
                self._workload['process_pool']['history'].append({
                    'timestamp': current_time,
                    'active_tasks': process_active,
                    'completed_tasks': self._workload['process_pool']['completed_tasks'],
                    'failed_tasks': self._workload['process_pool']['failed_tasks'],
                    'avg_completion_time': self._workload['process_pool']['avg_completion_time']
                })
                
                # Update peak metrics
                with self._locks['performance']:
                    self._performance['memory_peak'] = max(self._performance['memory_peak'], metrics['memory_usage'])
                    self._performance['cpu_peak'] = max(self._performance['cpu_peak'], metrics['cpu_usage'])
                
        except Exception as e:
            self.log.error(f"Error updating workload stats: {str(e)}\n{traceback.format_exc()}")
            
    def _adjust_resources(self):
        """Adjust resources based on workload and system metrics."""
        try:
            with self._locks['workload']:
                # Get current system metrics
                metrics = self._resource_monitor.get_metrics()
                cpu_usage = metrics['cpu_usage']
                memory_usage = metrics['memory_usage']
                
                # Get performance metrics
                thread_metrics = self._workload['thread_pool']['performance_metrics']
                process_metrics = self._workload['process_pool']['performance_metrics']
                
                # Calculate average metrics
                thread_throughput = statistics.mean(thread_metrics['throughput']) if thread_metrics['throughput'] else 0
                thread_latency = statistics.mean(thread_metrics['latency']) if thread_metrics['latency'] else 0
                thread_error_rate = statistics.mean(thread_metrics['error_rate']) if thread_metrics['error_rate'] else 0
                
                process_throughput = statistics.mean(process_metrics['throughput']) if process_metrics['throughput'] else 0
                process_latency = statistics.mean(process_metrics['latency']) if process_metrics['latency'] else 0
                process_error_rate = statistics.mean(process_metrics['error_rate']) if process_metrics['error_rate'] else 0
                
                # Adjust chunk sizes based on performance
                self._adjust_chunk_sizes(
                    thread_throughput, thread_latency, thread_error_rate,
                    process_throughput, process_latency, process_error_rate
                )
                
                # Adjust pool sizes based on system load
                if cpu_usage > 80 or memory_usage > 80:
                    # Reduce pool sizes under high load
                    self._reduce_pool_sizes()
                elif cpu_usage < 50 and memory_usage < 50:
                    # Increase pool sizes under low load
                    self._increase_pool_sizes()
                    
                # Clean up expired cache entries
                self._cleanup_cache()
                    
        except Exception as e:
            self.log.error(f"Error adjusting resources: {str(e)}\n{traceback.format_exc()}")
            
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            with self._locks['cache']:
                current_time = time.time()
                for level in ['l1', 'l2', 'l3']:
                    # Remove expired entries
                    expired_keys = [
                        key for key, timestamp in self._cache_timestamps[level].items()
                        if current_time - timestamp > self._cache_ttl[level]
                    ]
                    for key in expired_keys:
                        del self._cache[level][key]
                        del self._cache_timestamps[level][key]
                        
                    # Remove excess entries if cache is too large
                    while len(self._cache[level]) > self._cache_sizes[level]:
                        # Remove oldest entries first
                        oldest_key = min(
                            self._cache_timestamps[level].items(),
                            key=lambda x: x[1]
                        )[0]
                        del self._cache[level][oldest_key]
                        del self._cache_timestamps[level][oldest_key]
                        
        except Exception as e:
            self.log.error(f"Error cleaning up cache: {str(e)}\n{traceback.format_exc()}")
            
    def _adjust_chunk_sizes(self, thread_throughput: float, thread_latency: float, thread_error_rate: float,
                           process_throughput: float, process_latency: float, process_error_rate: float):
        """Adjust chunk sizes based on performance metrics."""
        try:
            with self._locks['chunk_sizes']:
                # Adjust thread pool chunk size
                if thread_error_rate > 0.1:  # High error rate
                    self._chunk_sizes['thread'] = max(10, self._chunk_sizes['thread'] // 2)
                elif thread_latency > 1.0:  # High latency
                    self._chunk_sizes['thread'] = max(10, self._chunk_sizes['thread'] // 2)
                elif thread_throughput < 10:  # Low throughput
                    self._chunk_sizes['thread'] = min(10000, self._chunk_sizes['thread'] * 2)
                    
                # Adjust process pool chunk size
                if process_error_rate > 0.1:  # High error rate
                    self._chunk_sizes['process'] = max(100, self._chunk_sizes['process'] // 2)
                elif process_latency > 2.0:  # High latency
                    self._chunk_sizes['process'] = max(100, self._chunk_sizes['process'] // 2)
                elif process_throughput < 5:  # Low throughput
                    self._chunk_sizes['process'] = min(10000, self._chunk_sizes['process'] * 2)
                    
        except Exception as e:
            self.log.error(f"Error adjusting chunk sizes: {str(e)}\n{traceback.format_exc()}")
            
    def _reduce_pool_sizes(self):
        """Reduce pool sizes under high load."""
        try:
            # Reduce thread pool size
            current_thread_workers = self.thread_pool._max_workers
            new_thread_workers = max(4, current_thread_workers // 2)
            if new_thread_workers != current_thread_workers:
                self.thread_pool._max_workers = new_thread_workers
                self.log.info(f"Reduced thread pool size to {new_thread_workers}")
                
            # Reduce process pool size
            current_process_workers = self.process_pool._max_workers
            new_process_workers = max(2, current_process_workers // 2)
            if new_process_workers != current_process_workers:
                self.process_pool._max_workers = new_process_workers
                self.log.info(f"Reduced process pool size to {new_process_workers}")
                
        except Exception as e:
            self.log.error(f"Error reducing pool sizes: {str(e)}\n{traceback.format_exc()}")
            
    def _increase_pool_sizes(self):
        """Increase pool sizes under low load."""
        try:
            # Get system resources
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Increase thread pool size
            current_thread_workers = self.thread_pool._max_workers
            max_thread_workers = min(cpu_count * 4, int(memory_gb * 2), 100)
            new_thread_workers = min(max_thread_workers, current_thread_workers * 2)
            if new_thread_workers != current_thread_workers:
                self.thread_pool._max_workers = new_thread_workers
                self.log.info(f"Increased thread pool size to {new_thread_workers}")
                
            # Increase process pool size
            current_process_workers = self.process_pool._max_workers
            max_process_workers = min(cpu_count, int(memory_gb), 20)
            new_process_workers = min(max_process_workers, current_process_workers * 2)
            if new_process_workers != current_process_workers:
                self.process_pool._max_workers = new_process_workers
                self.log.info(f"Increased process pool size to {new_process_workers}")
                
        except Exception as e:
            self.log.error(f"Error increasing pool sizes: {str(e)}\n{traceback.format_exc()}")
            
    def process_in_parallel(self, data: List[Any], func: Callable, pool_type: str = 'thread',
                          priority: TaskPriority = TaskPriority.NORMAL, chunk_size: Optional[int] = None) -> List[Any]:
        """Process data in parallel with optimized chunking and workload balancing."""
        try:
            # Select appropriate pool
            pool = self.thread_pool if pool_type == 'thread' else self.process_pool
            
            # Get optimal chunk size
            if chunk_size is None:
                with self._locks['chunk_sizes']:
                    chunk_size = self._chunk_sizes[pool_type]
                    
            # Split data into chunks
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            # Process chunks in parallel with retry mechanism
            futures = []
            for chunk in chunks:
                task = Task(
                    priority=priority,
                    func=func,
                    data=chunk,
                    timestamp=time.time()
                )
                
                # Submit task to priority queue
                self._task_queue.put((priority.value, task))
                
                # Get task from queue and submit to pool
                while not self._task_queue.empty():
                    _, task = self._task_queue.get()
                    future = pool.submit(self._process_with_retry, task)
                    futures.append(future)
                    
            # Collect results with timeout
            results = []
            for future in as_completed(futures, timeout=300):  # 5-minute timeout
                try:
                    result = future.result()
                    results.extend(result if isinstance(result, list) else [result])
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['completed_tasks'] += 1
                        else:
                            self._workload['process_pool']['completed_tasks'] += 1
                            
                except TimeoutError:
                    self.log.error("Task timed out")
                    future.cancel()
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
                except CancelledError:
                    self.log.warning("Task was cancelled")
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
                except Exception as e:
                    self.log.error(f"Error processing chunk: {str(e)}\n{traceback.format_exc()}")
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
            return results
            
        except Exception as e:
            self.log.error(f"Error processing in parallel: {str(e)}\n{traceback.format_exc()}")
            return []
            
    def _process_with_retry(self, task: Task) -> Any:
        """Process task with retry mechanism."""
        try:
            # Check task dependencies
            if task.dependencies:
                with self._locks['workload']:
                    if not all(dep in self._completed_tasks for dep in task.dependencies):
                        # Dependencies not met, requeue task
                        self._task_queue.put((task.priority.value, task))
                        return None
                        
            # Process task
            result = task.func(task.data)
            
            # Update performance metrics
            with self._locks['performance']:
                self._performance['successful_tasks'] += 1
                self._performance['total_processing_time'] += time.time() - task.timestamp
                
            # Mark task as completed
            with self._locks['workload']:
                self._completed_tasks.add(task.task_id)
                
            return result
            
        except Exception as e:
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                time.sleep(0.1 * task.retry_count)  # Exponential backoff
                return self._process_with_retry(task)
            else:
                # Update performance metrics
                with self._locks['performance']:
                    self._performance['failed_tasks'] += 1
                raise
                
    def process_stream(self, data_stream: Generator[Any, None, None], func: Callable,
                      pool_type: str = 'thread', priority: TaskPriority = TaskPriority.NORMAL,
                      chunk_size: Optional[int] = None) -> Generator[Any, None, None]:
        """Process data stream in parallel with optimized chunking and backpressure."""
        try:
            # Select appropriate pool
            pool = self.thread_pool if pool_type == 'thread' else self.process_pool
            
            # Get optimal chunk size
            if chunk_size is None:
                with self._locks['chunk_sizes']:
                    chunk_size = self._chunk_sizes[pool_type]
                    
            # Process stream in chunks with backpressure
            current_chunk = []
            futures = []
            max_futures = pool._max_workers * 2  # Limit concurrent futures
            
            for item in data_stream:
                current_chunk.append(item)
                
                if len(current_chunk) >= chunk_size:
                    # Check backpressure
                    while len(futures) >= max_futures:
                        # Process completed futures
                        for completed_future in as_completed(futures, timeout=0.1):
                            try:
                                result = completed_future.result()
                                yield from (result if isinstance(result, list) else [result])
                                
                                # Update workload stats
                                with self._locks['workload']:
                                    if pool_type == 'thread':
                                        self._workload['thread_pool']['completed_tasks'] += 1
                                    else:
                                        self._workload['process_pool']['completed_tasks'] += 1
                                        
                                futures.remove(completed_future)
                                break
                                
                            except TimeoutError:
                                self.log.error("Task timed out")
                                completed_future.cancel()
                                
                                # Update workload stats
                                with self._locks['workload']:
                                    if pool_type == 'thread':
                                        self._workload['thread_pool']['failed_tasks'] += 1
                                    else:
                                        self._workload['process_pool']['failed_tasks'] += 1
                                        
                                futures.remove(completed_future)
                                break
                                
                            except CancelledError:
                                self.log.warning("Task was cancelled")
                                
                                # Update workload stats
                                with self._locks['workload']:
                                    if pool_type == 'thread':
                                        self._workload['thread_pool']['failed_tasks'] += 1
                                    else:
                                        self._workload['process_pool']['failed_tasks'] += 1
                                        
                                futures.remove(completed_future)
                                break
                                
                            except Exception as e:
                                self.log.error(f"Error processing chunk: {str(e)}\n{traceback.format_exc()}")
                                
                                # Update workload stats
                                with self._locks['workload']:
                                    if pool_type == 'thread':
                                        self._workload['thread_pool']['failed_tasks'] += 1
                                    else:
                                        self._workload['process_pool']['failed_tasks'] += 1
                                        
                                futures.remove(completed_future)
                                break
                                
                    # Submit new chunk
                    task = Task(
                        priority=priority,
                        func=func,
                        data=current_chunk,
                        timestamp=time.time()
                    )
                    future = pool.submit(self._process_with_retry, task)
                    futures.append(future)
                    current_chunk = []
                    
            # Process remaining items
            if current_chunk:
                task = Task(
                    priority=priority,
                    func=func,
                    data=current_chunk,
                    timestamp=time.time()
                )
                future = pool.submit(self._process_with_retry, task)
                futures.append(future)
                
            # Process remaining futures
            for future in as_completed(futures):
                try:
                    result = future.result()
                    yield from (result if isinstance(result, list) else [result])
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['completed_tasks'] += 1
                        else:
                            self._workload['process_pool']['completed_tasks'] += 1
                            
                except TimeoutError:
                    self.log.error("Task timed out")
                    future.cancel()
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
                except CancelledError:
                    self.log.warning("Task was cancelled")
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
                except Exception as e:
                    self.log.error(f"Error processing final chunk: {str(e)}\n{traceback.format_exc()}")
                    
                    # Update workload stats
                    with self._locks['workload']:
                        if pool_type == 'thread':
                            self._workload['thread_pool']['failed_tasks'] += 1
                        else:
                            self._workload['process_pool']['failed_tasks'] += 1
                            
        except Exception as e:
            self.log.error(f"Error processing stream: {str(e)}\n{traceback.format_exc()}")
            yield from []
            
    def get_workload_stats(self) -> Dict:
        """Get detailed workload statistics."""
        try:
            with self._locks['workload']:
                return {
                    'thread_pool': self._workload['thread_pool'].copy(),
                    'process_pool': self._workload['process_pool'].copy(),
                    'chunk_sizes': self._chunk_sizes.copy(),
                    'performance': self._performance.copy(),
                    'resources': self._resource_monitor.get_metrics()
                }
                
        except Exception as e:
            self.log.error(f"Error getting workload stats: {str(e)}\n{traceback.format_exc()}")
            return {}
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop resource monitoring
            self._resource_monitor.stop()
            
            with self._locks['workload']:
                # Shutdown pools
                self.thread_pool.shutdown(wait=True)
                self.process_pool.shutdown(wait=True)
                
                # Clear task queue
                while not self._task_queue.empty():
                    try:
                        self._task_queue.get_nowait()
                    except Empty:
                        break
                        
                # Clear workload tracking
                self._workload['thread_pool']['history'].clear()
                self._workload['process_pool']['history'].clear()
                
                # Clear performance metrics
                for metrics in self._workload['thread_pool']['performance_metrics'].values():
                    metrics.clear()
                for metrics in self._workload['process_pool']['performance_metrics'].values():
                    metrics.clear()
                    
            with self._locks['cache']:
                # Clear cache
                for cache_level in self._cache.values():
                    cache_level.clear()
                for timestamps in self._cache_timestamps.values():
                    timestamps.clear()
                    
            # Clear task tracking
            self._task_dependencies.clear()
            self._completed_tasks.clear()
                    
            # Force garbage collection
            gc.collect(2)
            
        except Exception as e:
            self.log.error(f"Error during cleanup: {str(e)}\n{traceback.format_exc()}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 