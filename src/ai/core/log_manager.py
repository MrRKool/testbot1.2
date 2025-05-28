import logging
from typing import Dict, List, Any, Optional
import os
from datetime import datetime, timedelta
import json
import yaml
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class LogManager:
    """Manages logging for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Log settings
        self.log_dir = config.get('log_dir', 'logs')
        self.max_log_size = config.get('max_log_size_mb', 10) * 1024 * 1024  # Convert to bytes
        self.max_log_files = config.get('max_log_files', 5)
        self.log_format = config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_level = getattr(logging, config.get('log_level', 'INFO'))
        
        # Initialize loggers
        self.loggers = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Setup root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.log_level)
            
            # Clear existing handlers
            root_logger.handlers = []
            
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.log_format))
            root_logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, 'ai.log'),
                maxBytes=self.max_log_size,
                backupCount=self.max_log_files
            )
            file_handler.setFormatter(logging.Formatter(self.log_format))
            root_logger.addHandler(file_handler)
            
            # Setup component loggers
            for component in self.config.get('components', []):
                self._setup_component_logger(component)
                
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            
    def _setup_component_logger(self, component: str):
        """Setup logger for a specific component."""
        try:
            # Create component logger
            logger = logging.getLogger(component)
            logger.setLevel(self.log_level)
            
            # Add file handler
            file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, f"{component}.log"),
                maxBytes=self.max_log_size,
                backupCount=self.max_log_files
            )
            file_handler.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(file_handler)
            
            # Store logger
            self.loggers[component] = logger
            
        except Exception as e:
            self.logger.error(f"Error setting up logger for {component}: {str(e)}")
            
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for a component."""
        try:
            if component not in self.loggers:
                self._setup_component_logger(component)
            return self.loggers[component]
            
        except Exception as e:
            self.logger.error(f"Error getting logger for {component}: {str(e)}")
            return logging.getLogger(component)
            
    def log_event(self, component: str, event_type: str, message: str, data: Dict = None):
        """Log an event."""
        try:
            logger = self.get_logger(component)
            
            # Create event
            event = {
                'timestamp': datetime.now().isoformat(),
                'component': component,
                'type': event_type,
                'message': message,
                'data': data or {}
            }
            
            # Log based on event type
            if event_type == 'error':
                logger.error(json.dumps(event))
            elif event_type == 'warning':
                logger.warning(json.dumps(event))
            elif event_type == 'info':
                logger.info(json.dumps(event))
            elif event_type == 'debug':
                logger.debug(json.dumps(event))
            else:
                logger.info(json.dumps(event))
                
        except Exception as e:
            self.logger.error(f"Error logging event: {str(e)}")
            
    def get_log_stats(self, component: str = None) -> Dict:
        """Get statistics about logs."""
        try:
            stats = {
                'total_size': 0,
                'file_count': 0,
                'oldest_log': None,
                'newest_log': None
            }
            
            # Get log files
            log_files = []
            if component:
                log_files = [f for f in os.listdir(self.log_dir) if f.startswith(component)]
            else:
                log_files = os.listdir(self.log_dir)
                
            if not log_files:
                return stats
                
            # Calculate statistics
            for filename in log_files:
                filepath = os.path.join(self.log_dir, filename)
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                stats['total_size'] += size
                stats['file_count'] += 1
                
                if not stats['oldest_log'] or mtime < stats['oldest_log']:
                    stats['oldest_log'] = mtime
                if not stats['newest_log'] or mtime > stats['newest_log']:
                    stats['newest_log'] = mtime
                    
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting log stats: {str(e)}")
            return {}
            
    def cleanup_old_logs(self, max_age: timedelta = timedelta(days=30)):
        """Remove old log files."""
        try:
            if not os.path.exists(self.log_dir):
                return
                
            # Get current time
            now = datetime.now()
            
            # Check each log file
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(self.log_dir, filename)
                    file_age = now - datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    # Remove old files
                    if file_age > max_age:
                        os.remove(filepath)
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {str(e)}")
            
    def export_logs(self, component: str = None, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Export logs for analysis."""
        try:
            logs = []
            
            # Get log files
            log_files = []
            if component:
                log_files = [f for f in os.listdir(self.log_dir) if f.startswith(component)]
            else:
                log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
                
            # Read logs
            for filename in log_files:
                filepath = os.path.join(self.log_dir, filename)
                
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            # Parse log entry
                            log_entry = json.loads(line)
                            timestamp = datetime.fromisoformat(log_entry['timestamp'])
                            
                            # Apply filters
                            if start_date and timestamp < start_date:
                                continue
                            if end_date and timestamp > end_date:
                                continue
                                
                            logs.append(log_entry)
                            
                        except json.JSONDecodeError:
                            continue
                            
            return logs
            
        except Exception as e:
            self.logger.error(f"Error exporting logs: {str(e)}")
            return [] 