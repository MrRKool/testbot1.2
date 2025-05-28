import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import psutil
import os
from dataclasses import dataclass
from queue import Queue
import telegram
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MonitorConfig:
    """Configuratie voor monitoring."""
    log_dir: str = "logs"
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    alert_thresholds: Dict[str, float] = None
    check_interval: int = 60  # seconden
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class Monitor:
    """Monitor systeem en trading performance."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or MonitorConfig()
        
        # Maak log directory
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Telegram setup
        self.telegram_bot = None
        if self.config.telegram_token and self.config.telegram_chat_id:
            self.telegram_bot = telegram.Bot(token=self.config.telegram_token)
            
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.alert_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance metrics
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_io": [],
            "api_errors": 0,
            "trade_count": 0,
            "error_count": 0
        }
        
    def _setup_logging(self):
        """Configureer logging."""
        # Maak log bestand
        log_file = self.log_dir / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configureer file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Configureer console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Voeg handlers toe
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        
    def start_monitoring(self):
        """Start de monitoring thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Start alert processor
            self.executor.submit(self._process_alerts)
            
            self.logger.info("Monitoring gestart")
            
    def stop_monitoring(self):
        """Stop de monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.executor.shutdown()
        self.logger.info("Monitoring gestopt")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Verzamel metrics
                self._collect_metrics()
                
                # Check thresholds
                self._check_thresholds()
                
                # Log metrics
                self._log_metrics()
                
                # Wacht voor volgende check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Fout in monitoring loop: {e}")
                
    def _collect_metrics(self):
        """Verzamel systeem metrics."""
        # CPU usage
        self.metrics["cpu_usage"].append(psutil.cpu_percent())
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics["memory_usage"].append(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics["disk_usage"].append(disk.percent)
        
        # Network IO
        net_io = psutil.net_io_counters()
        self.metrics["network_io"].append({
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        })
        
    def _check_thresholds(self):
        """Check metrics tegen thresholds."""
        if not self.config.alert_thresholds:
            return
            
        # CPU threshold
        if self.metrics["cpu_usage"][-1] > self.config.alert_thresholds.get("cpu", 90):
            self._queue_alert("CPU usage te hoog", "warning")
            
        # Memory threshold
        if self.metrics["memory_usage"][-1] > self.config.alert_thresholds.get("memory", 90):
            self._queue_alert("Memory usage te hoog", "warning")
            
        # Disk threshold
        if self.metrics["disk_usage"][-1] > self.config.alert_thresholds.get("disk", 90):
            self._queue_alert("Disk usage te hoog", "warning")
            
    def _log_metrics(self):
        """Log huidige metrics."""
        metrics_str = (
            f"CPU: {self.metrics['cpu_usage'][-1]}% | "
            f"Memory: {self.metrics['memory_usage'][-1]}% | "
            f"Disk: {self.metrics['disk_usage'][-1]}% | "
            f"API Errors: {self.metrics['api_errors']} | "
            f"Trades: {self.metrics['trade_count']}"
        )
        self.logger.info(metrics_str)
        
    def _queue_alert(self, message: str, level: str = "info"):
        """Voeg alert toe aan queue."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level
        }
        self.alert_queue.put(alert)
        
    def _process_alerts(self):
        """Verwerk alerts uit de queue."""
        while self.monitoring:
            try:
                alert = self.alert_queue.get()
                self._send_alert(alert)
                self.alert_queue.task_done()
            except Exception as e:
                self.logger.error(f"Fout bij verwerken alert: {e}")
                
    def _send_alert(self, alert: Dict[str, Any]):
        """Stuur alert via Telegram."""
        if not self.telegram_bot:
            return
            
        try:
            # Format message
            message = (
                f"🚨 {alert['level'].upper()} Alert\n"
                f"⏰ {alert['timestamp']}\n"
                f"📝 {alert['message']}"
            )
            
            # Stuur via Telegram
            asyncio.run(self.telegram_bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message
            ))
            
        except Exception as e:
            self.logger.error(f"Fout bij sturen Telegram alert: {e}")
            
    def record_api_error(self):
        """Record een API error."""
        self.metrics["api_errors"] += 1
        self._queue_alert("API error opgetreden", "error")
        
    def record_trade(self):
        """Record een trade."""
        self.metrics["trade_count"] += 1
        
    def record_error(self):
        """Record een error."""
        self.metrics["error_count"] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Haal huidige metrics op."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": self.metrics["cpu_usage"][-1] if self.metrics["cpu_usage"] else 0,
            "memory_usage": self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else 0,
            "disk_usage": self.metrics["disk_usage"][-1] if self.metrics["disk_usage"] else 0,
            "api_errors": self.metrics["api_errors"],
            "trade_count": self.metrics["trade_count"],
            "error_count": self.metrics["error_count"]
        }
        
    def save_metrics(self, filepath: str = "metrics.json"):
        """Sla metrics op."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Fout bij opslaan metrics: {e}")
            
    def load_metrics(self, filepath: str = "metrics.json"):
        """Laad metrics."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.metrics = json.load(f)
        except Exception as e:
            self.logger.error(f"Fout bij laden metrics: {e}")
            
    def cleanup_old_logs(self):
        """Ruim oude log bestanden op."""
        try:
            log_files = sorted(self.log_dir.glob("monitor_*.log"))
            while len(log_files) > self.config.backup_count:
                oldest_file = log_files.pop(0)
                oldest_file.unlink()
                self.logger.info(f"Oud log bestand verwijderd: {oldest_file}")
        except Exception as e:
            self.logger.error(f"Fout bij opschonen logs: {e}") 