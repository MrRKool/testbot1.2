import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import logging
import logging.handlers
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv('testbot1.env')

@dataclass
class SymbolConfig:
    """Configuration for a trading symbol."""
    min_volume: float
    min_volatility: float
    max_spread: float
    leverage: int
    enabled: bool = True
    max_position_size: Optional[float] = None
    min_position_size: Optional[float] = None

@dataclass
class RiskConfig:
    """Risk management configuration."""
    stop_loss: Dict[str, Any]
    take_profit: Dict[str, Any]
    trailing_stop: Dict[str, Any]
    max_daily_loss: float
    max_drawdown: float
    min_risk_reward: float
    base_risk_per_trade: float
    max_open_trades: int
    max_daily_trades: int

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float
    commission: float
    slippage: float
    risk_free_rate: float = 0.02
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str
    testnet: bool
    api_key: str
    api_secret: str
    base_url: str
    timeout: int
    max_retries: int
    retry_delay: int
    rate_limit: Dict[str, int]

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str
    max_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class BotConfig:
    """Main configuration for the trading bot."""
    symbols: Dict[str, SymbolConfig]
    timeframes: List[str]
    update_interval: int
    polling_interval: int
    report_interval: int
    risk: RiskConfig
    backtest: BacktestConfig
    exchange: ExchangeConfig
    logging: LoggingConfig

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Exception for configuration validation errors."""
    pass

class ConfigLoadError(ConfigError):
    """Exception for configuration loading errors."""
    pass

# Default configuration
DEFAULTS = {
    "trading": {
        "symbols": {
            "BTCUSDT": {
                "min_volume": 1000000,
                "min_volatility": 0.001,
                "max_spread": 0.001,
                "leverage": 1,
                "enabled": True,
                "max_position_size": None,
                "min_position_size": None
            },
            "ETHUSDT": {
                "min_volume": 500000,
                "min_volatility": 0.001,
                "max_spread": 0.001,
                "leverage": 1,
                "enabled": True,
                "max_position_size": None,
                "min_position_size": None
            }
        },
        "timeframes": ["1m", "5m", "15m", "1h"],
        "update_interval": 1,
        "polling_interval": 1,
        "report_interval": 3600
    },
    "risk": {
        "stop_loss": {
            "type": "fixed",
            "value": 0.015  # 1.5%
        },
        "take_profit": {
            "type": "fixed",
            "value": 0.03  # 3%
        },
        "trailing_stop": {
            "enabled": True,
            "activation": 0.01,  # 1%
            "distance": 0.005  # 0.5%
        },
        "max_daily_loss": 0.02,
        "max_drawdown": 0.15,
        "min_risk_reward": 2.0,
        "base_risk_per_trade": 0.01,
        "max_open_trades": 3,
        "max_daily_trades": 10
    },
    "backtest": {
        "initial_capital": 10000,
        "commission": 0.001,  # 0.1%
        "slippage": 0.001,  # 0.1%
        "risk_free_rate": 0.02
    },
    "exchange": {
        "name": "bybit",
        "testnet": True,
        "api_key": os.getenv("BYBIT_API_KEY", ""),
        "api_secret": os.getenv("BYBIT_API_SECRET", ""),
        "base_url": "https://api-testnet.bybit.com",
        "timeout": 10,
        "max_retries": 3,
        "retry_delay": 5,
        "rate_limit": {
            "calls": 2,
            "period": 1
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/trading_bot.log",
        "max_size": 10485760,  # 10MB
        "backup_count": 5
    }
}

class ConfigManager:
    """Manages configuration loading, validation, and caching."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration manager."""
        self.config_dir = config_dir
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Any] = {}
        self._last_modified: Dict[str, float] = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging."""
        try:
            log_path = Path("logs")
            log_path.mkdir(parents=True, exist_ok=True)
            
            handlers = [
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    log_path / "config.log",
                    maxBytes=10485760,  # 10MB
                    backupCount=5,
                    encoding="utf-8"
                )
            ]
            
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=handlers
            )
            
        except Exception as e:
            raise ConfigError(f"Error setting up logging: {e}")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values."""
        try:
            errors = []
            warnings = []
            
            # Validate trading section
            if "trading" not in config:
                errors.append("Missing trading section in config")
            else:
                # Validate symbols
                if "symbols" not in config["trading"]:
                    errors.append("Missing symbols in trading config")
                else:
                    for symbol, settings in config["trading"]["symbols"].items():
                        required_fields = ["min_volume", "min_volatility", "max_spread", "leverage"]
                        for field in required_fields:
                            if field not in settings:
                                errors.append(f"Missing {field} in {symbol} config")
                
                # Validate timeframes
                if "timeframes" not in config["trading"]:
                    errors.append("Missing timeframes in trading config")
                else:
                    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                    for tf in config["trading"]["timeframes"]:
                        if tf not in valid_timeframes:
                            errors.append(f"Invalid timeframe: {tf}")
            
            # Validate risk section
            if "risk" not in config:
                errors.append("Missing risk section in config")
            else:
                required_risk_fields = [
                    "max_daily_loss", "max_drawdown", "min_risk_reward",
                    "base_risk_per_trade", "max_open_trades", "max_daily_trades"
                ]
                for field in required_risk_fields:
                    if field not in config["risk"]:
                        errors.append(f"Missing {field} in risk config")
            
            # Validate exchange section
            if "exchange" not in config:
                errors.append("Missing exchange section in config")
            else:
                required_exchange_fields = [
                    "name", "testnet", "api_key", "api_secret", "base_url",
                    "timeout", "max_retries", "retry_delay", "rate_limit"
                ]
                for field in required_exchange_fields:
                    if field not in config["exchange"]:
                        errors.append(f"Missing {field} in exchange config")
            
            # Log results
            if errors:
                for error in errors:
                    self.logger.error(f"Configuration error: {error}")
                return False
                
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"Configuration warning: {warning}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def load_config(self, config_path: str = "config/config.yaml") -> Dict[str, Any]:
        """Load and validate configuration from file."""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Check cache
            if self._is_cache_valid(config_path):
                self.logger.debug(f"Using cached config from {config_path}")
                return self._cache[config_path]
            
            # Load config from file if it exists
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file not found at {config_path}, using defaults")
                config = DEFAULTS
            
            # Validate config
            if not self.validate_config(config):
                self.logger.error("Invalid configuration, using defaults")
                config = DEFAULTS
            
            # Update cache
            self._update_cache(config_path, config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            self.logger.info("Using default configuration")
            return DEFAULTS

    def save_config(self, config: Dict[str, Any], config_path: str = "config/config.yaml"):
        """Save configuration to file."""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Validate config before saving
            if not self.validate_config(config):
                raise ConfigValidationError("Invalid configuration")
            
            # Save config to file
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update cache
            self._update_cache(config_path, config)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config to {config_path}: {e}")
            raise

    def _is_cache_valid(self, config_path: str) -> bool:
        """Check if cached configuration is still valid."""
        if config_path not in self._cache or config_path not in self._last_modified:
            return False
        
        try:
            current_mtime = os.path.getmtime(config_path)
            return current_mtime == self._last_modified[config_path]
        except:
            return False

    def _update_cache(self, config_path: str, config_data: Dict[str, Any]) -> None:
        """Update configuration cache."""
        try:
            self._cache[config_path] = config_data
            self._last_modified[config_path] = os.path.getmtime(config_path)
        except:
            pass

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()
        self._last_modified.clear()

# Create global config manager instance
config_manager = ConfigManager()

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration using the global config manager."""
    return config_manager.load_config(config_path)

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml"):
    """Save configuration using the global config manager."""
    config_manager.save_config(config, config_path)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration using the global config manager."""
    return config_manager.validate_config(config) 