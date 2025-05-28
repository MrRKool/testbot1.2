import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Configuration loader for the trading bot."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the configuration loader."""
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.config = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self._validate_config()
            self.logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        required_sections = [
            'api',
            'trading',
            'monitoring',
            'database',
            'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate API configuration
        api_config = self.config['api']
        required_api_fields = ['api_key', 'api_secret', 'base_url']
        for field in required_api_fields:
            if field not in api_config:
                raise ValueError(f"Missing required API configuration field: {field}")
        
        # Validate trading configuration
        trading_config = self.config['trading']
        required_trading_fields = ['symbols', 'timeframe', 'strategy']
        for field in required_trading_fields:
            if field not in trading_config:
                raise ValueError(f"Missing required trading configuration field: {field}")
        
        # Validate monitoring configuration
        monitoring_config = self.config['monitoring']
        required_monitoring_fields = ['check_interval', 'alert_thresholds']
        for field in required_monitoring_fields:
            if field not in monitoring_config:
                raise ValueError(f"Missing required monitoring configuration field: {field}")
        
        # Validate database configuration
        db_config = self.config['database']
        required_db_fields = ['url', 'pool_size', 'max_overflow']
        for field in required_db_fields:
            if field not in db_config:
                raise ValueError(f"Missing required database configuration field: {field}")
        
        # Validate logging configuration
        logging_config = self.config['logging']
        required_logging_fields = ['level', 'format', 'file']
        for field in required_logging_fields:
            if field not in logging_config:
                raise ValueError(f"Missing required logging configuration field: {field}")

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Helper function to load configuration."""
    loader = ConfigLoader(config_path)
    return loader.load_config() 