#!/usr/bin/env python3
import os
import sys
import time
import logging
import requests
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import yaml
import platform
import threading
from queue import Queue
import pandas as pd
import numpy as np
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging.handlers
import signal
import subprocess
import gc
from logging.handlers import RotatingFileHandler
from utils.system_service import ServiceManager

def setup_logging():
    """Setup logging with proper rotation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler met rotatie
    file_handler = RotatingFileHandler(
        filename=str(log_dir / "monitor.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    bot_running: bool
    last_check: datetime
    api_latency: List[float] = None
    error_count: int = 0
    trade_count: int = 0
    consecutive_failures: int = 0
    last_alert_time: datetime = None
    
    def __post_init__(self):
        self.api_latency = []
        self.last_alert_time = datetime.now()
        self.last_check = datetime.now()

class BotMonitor:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.logger = setup_logging()
        self.config_path = config_path
        self.last_check = datetime.now()
        self.performance_metrics = PerformanceMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io={'bytes_sent': 0.0, 'bytes_recv': 0.0},
            bot_running=False,
            last_check=datetime.now()
        )
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 80,
            'disk_percent': 90,
            'log_size_mb': 100,
            'error_count': 10,
            'restart_count': 5,
            'api_latency_ms': 1000,
            'memory_leak_mb': 100,
        }
        
        # Initialize thread pool with optimal size
        self.executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))
        
        # Initialize alert queue with max size
        self.alert_queue = Queue(maxsize=100)
        
        # Load config
        self._load_config()
        
        # Platform specific settings
        self.is_macos = platform.system() == 'Darwin'
        
        # Alert cooldown period (5 minutes)
        self.alert_cooldown = timedelta(minutes=5)
        
        # Cache for system metrics
        self._cpu_percent = psutil.cpu_percent()
        self._memory = psutil.virtual_memory()
        self._disk = psutil.disk_usage('/')
        self._net_io = psutil.net_io_counters()
        
    def _load_config(self):
        """Load configuration from file with error handling."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = {}
            
    def check_system_resources(self):
        """Check system resources with caching."""
        try:
            # Update cached values
            self._cpu_percent = psutil.cpu_percent(interval=1)
            self._memory = psutil.virtual_memory()
            self._disk = psutil.disk_usage('/')
            self._net_io = psutil.net_io_counters()
            
            # Update metrics
            self.performance_metrics.cpu_usage = self._cpu_percent
            self.performance_metrics.memory_usage = self._memory.percent
            self.performance_metrics.disk_usage = self._disk.percent
            self.performance_metrics.network_io = {
                'bytes_sent': self._net_io.bytes_sent,
                'bytes_recv': self._net_io.bytes_recv
            }
            
            # Check for potential issues
            if self.performance_metrics.cpu_usage > 90:
                self.logger.warning(f"High CPU usage: {self.performance_metrics.cpu_usage}%")
            if self.performance_metrics.memory_usage > 90:
                self.logger.warning(f"High memory usage: {self.performance_metrics.memory_usage}%")
            if self.performance_metrics.disk_usage > 90:
                self.logger.warning(f"High disk usage: {self.performance_metrics.disk_usage}%")
                
        except Exception as e:
            self.logger.error(f"Error checking system resources: {str(e)}")
            
    def check_bot_process(self):
        """Check if bot process is running with improved error handling."""
        try:
            service_manager = ServiceManager("trading_bot")
            
            # Check PID file first
            if service_manager.pid_file.exists():
                try:
                    pid = int(service_manager.pid_file.read_text().strip())
                    process = psutil.Process(pid)
                    if process.is_running() and process.name().lower().startswith('python'):
                        self.performance_metrics.bot_running = True
                        self.logger.info(f"Bot process running with PID {pid}")
                        return True
                except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.debug(f"PID check failed: {str(e)}")
            
            # Check launchctl for macOS
            if service_manager.is_macos:
                try:
                    result = subprocess.run(
                        ['launchctl', 'list', f'com.{service_manager.service_name}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and '0' in result.stdout:
                        self.performance_metrics.bot_running = True
                        self.logger.info("Bot service is running via launchctl")
                        return True
                except subprocess.TimeoutExpired:
                    self.logger.error("Timeout checking launchctl status")
                except Exception as e:
                    self.logger.error(f"Error checking launchctl: {str(e)}")
            
            # Only log warning if cooldown period has passed
            current_time = datetime.now()
            if not self.performance_metrics.bot_running:
                if (current_time - self.performance_metrics.last_alert_time) > self.alert_cooldown:
                    self.logger.warning("Bot process is not running")
                    self.performance_metrics.last_alert_time = current_time
                    
                    # Try to restart the bot
                    try:
                        if service_manager.start_service():
                            self.logger.info("Successfully restarted bot service")
                            self.performance_metrics.bot_running = True
                            return True
                        else:
                            self.logger.error("Failed to restart bot service")
                    except Exception as e:
                        self.logger.error(f"Error restarting bot service: {str(e)}")
            
            return self.performance_metrics.bot_running
                
        except Exception as e:
            self.logger.error(f"Error in check_bot_process: {str(e)}")
            self.performance_metrics.bot_running = False
            return False
            
    async def check_api_health(self):
        """Check API health with optimized connection handling."""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    "https://api.bybit.com/v5/market/time",
                    timeout=aiohttp.ClientTimeout(total=5),
                    ssl=False,
                    headers={'User-Agent': 'TradingBot/1.0'}
                ) as response:
                    latency = (time.time() - start_time) * 1000
                    self.performance_metrics.api_latency.append(latency)
                    return {
                        'status': response.status,
                        'latency_ms': latency,
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            self.logger.error(f"Error checking API health: {str(e)}")
            return {
                'status': 'error',
                'latency_ms': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def send_alert(self, message: str, level: str = 'warning'):
        """Send alert with rate limiting and cooldown."""
        current_time = datetime.now()
        
        # Check cooldown period
        if (current_time - self.performance_metrics.last_alert_time) < self.alert_cooldown:
            return
            
        try:
            if 'telegram' in self.config and self.config['telegram']['enabled']:
                token = self.config['telegram']['token']
                chat_id = self.config['telegram']['chat_id']
                
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {
                    'chat_id': chat_id,
                    'text': f"🚨 {level.upper()}: {message}",
                    'parse_mode': 'HTML'
                }
                
                # Use session for connection pooling
                with requests.Session() as session:
                    for attempt in range(3):
                        try:
                            response = session.post(url, data=data, timeout=5)
                            if response.status_code == 200:
                                self.performance_metrics.last_alert_time = current_time
                                return
                            time.sleep(1)
                        except Exception as e:
                            self.logger.error(f"Failed to send Telegram alert (attempt {attempt+1}): {str(e)}")
                            time.sleep(1)
                            
                self.logger.error("Failed to send Telegram alert after 3 attempts")
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            
    async def run_health_check(self):
        """Run health check with optimized monitoring."""
        while True:
            try:
                # Check system resources
                self.check_system_resources()
                
                # Check bot process
                self.check_bot_process()
                
                # Check API health
                api_metrics = await self.check_api_health()
                
                # Handle bot process status
                if not self.performance_metrics.bot_running:
                    self.performance_metrics.consecutive_failures += 1
                    
                    # Only send alert if cooldown period has passed
                    if (datetime.now() - self.performance_metrics.last_alert_time) >= self.alert_cooldown:
                        self.send_alert(
                            f"Bot process is not running (attempt {self.performance_metrics.consecutive_failures}/3)",
                            level='warning'
                        )
                    
                    if self.performance_metrics.consecutive_failures >= 3:
                        self.logger.warning("Maximum consecutive failures reached, attempting to restart bot")
                        self.restart_service()
                        self.performance_metrics.consecutive_failures = 0
                else:
                    self.performance_metrics.consecutive_failures = 0
                
                # Check system metrics
                if self.performance_metrics.cpu_usage > self.alert_thresholds['cpu_percent']:
                    self.send_alert(f"High CPU usage: {self.performance_metrics.cpu_usage}%", level='warning')
                    
                if self.performance_metrics.memory_usage > self.alert_thresholds['memory_percent']:
                    self.send_alert(f"High memory usage: {self.performance_metrics.memory_usage}%", level='warning')
                
                if api_metrics.get('latency_ms', 0) > self.alert_thresholds['api_latency_ms']:
                    self.send_alert(f"High API latency: {api_metrics['latency_ms']}ms", level='warning')
                
                self.logger.info("Health check completed successfully")
                
                # Cleanup
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error in health check: {str(e)}")
                self.send_alert(f"Health check error: {str(e)}", level='error')
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    def restart_service(self):
        """Restart the bot service with improved error handling."""
        try:
            service_manager = ServiceManager("trading_bot")
            
            if self.is_macos:
                self.logger.info("Restarting bot service on macOS...")
                if self.check_bot_process():
                    try:
                        os.kill(psutil.Process(service_manager.pid).pid, signal.SIGTERM)
                        time.sleep(2)
                    except ProcessLookupError:
                        pass
                
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                subprocess.Popen(
                    ['python', 'main.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    start_new_session=True
                )
                self.logger.info("Bot service restarted successfully")
            else:
                try:
                    subprocess.run(['systemctl', 'restart', service_manager.service_name], 
                                 check=True,
                                 capture_output=True)
                    self.logger.info("Bot service restarted successfully")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to restart service: {e.stderr.decode()}")
                    raise
        except Exception as e:
            self.logger.error(f"Error restarting service: {str(e)}")
            raise

def main():
    setup_logging()
    service_manager = ServiceManager("trading_bot")
    
    logging.info("Starting monitor bot...")
    
    while True:
        try:
            # Controleer systeem resources
            resources = check_system_resources()
            logging.info(f"System resources: {json.dumps(resources, indent=2)}")
            
            # Controleer bot process
            if not check_bot_process():
                logging.warning("Bot process not running, attempting to restart...")
                try:
                    service_manager.start_service()
                    logging.info("Bot process restarted successfully")
                except Exception as e:
                    logging.error(f"Failed to restart bot process: {str(e)}")
            
            time.sleep(60)  # Wacht 60 seconden voor volgende check
            
        except Exception as e:
            logging.error(f"Error in monitor loop: {str(e)}")
            time.sleep(60)  # Wacht 60 seconden bij een error

if __name__ == "__main__":
    main() 