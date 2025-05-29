import unittest
from .config import (
    load_config,
    save_config,
    validate_config,
    DEFAULTS,
    ConfigManager,
    BotConfig,
    SymbolConfig,
    RiskConfig,
    BacktestConfig,
    ExchangeConfig,
    LoggingConfig
)
import os
import yaml
from pathlib import Path

class TestConfig(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_config_path = "config/test_config.yaml"
        self.config_manager = ConfigManager()
        
        # Create test config directory if it doesn't exist
        os.makedirs("config", exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
            
    def test_load_defaults(self):
        """Test loading default configuration."""
        config = load_config("nonexistent.yaml")
        self.assertEqual(config, DEFAULTS)
        
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        # Save test config
        test_config = {
            "trading": {
                "symbols": {
                    "BTCUSDT": {
                        "min_volume": 2000000,
                        "min_volatility": 0.002,
                        "max_spread": 0.002,
                        "leverage": 2,
                        "enabled": True
                    }
                },
                "timeframes": ["1m", "5m", "15m"],
                "update_interval": 2,
                "polling_interval": 2,
                "report_interval": 7200
            },
            "risk": DEFAULTS["risk"],
            "backtest": DEFAULTS["backtest"],
            "exchange": DEFAULTS["exchange"],
            "logging": DEFAULTS["logging"]
        }
        
        save_config(test_config, self.test_config_path)
        
        # Load and verify
        loaded_config = load_config(self.test_config_path)
        self.assertEqual(loaded_config["trading"]["symbols"]["BTCUSDT"]["min_volume"], 2000000)
        self.assertEqual(loaded_config["trading"]["timeframes"], ["1m", "5m", "15m"])
        
    def test_validate_config(self):
        """Test configuration validation."""
        # Test valid config
        self.assertTrue(validate_config(DEFAULTS))
        
        # Test invalid config
        invalid_config = DEFAULTS.copy()
        del invalid_config["trading"]
        self.assertFalse(validate_config(invalid_config))
        
    def test_symbol_config(self):
        """Test symbol configuration."""
        symbol_config = SymbolConfig(
            min_volume=1000000,
            min_volatility=0.001,
            max_spread=0.001,
            leverage=1
        )
        self.assertEqual(symbol_config.min_volume, 1000000)
        self.assertEqual(symbol_config.leverage, 1)
        
    def test_risk_config(self):
        """Test risk configuration."""
        risk_config = RiskConfig(
            stop_loss={"type": "fixed", "value": 0.02},
            take_profit={"type": "fixed", "value": 0.04},
            trailing_stop={"enabled": True, "activation": 0.01, "distance": 0.005},
            max_daily_loss=0.02,
            max_drawdown=0.15,
            min_risk_reward=2.0,
            base_risk_per_trade=0.01,
            max_open_trades=3,
            max_daily_trades=10
        )
        self.assertEqual(risk_config.max_daily_loss, 0.02)
        self.assertEqual(risk_config.max_open_trades, 3)
        
    def test_backtest_config(self):
        """Test backtest configuration."""
        backtest_config = BacktestConfig(
            initial_capital=10000,
            commission=0.001,
            slippage=0.001
        )
        self.assertEqual(backtest_config.initial_capital, 10000)
        self.assertEqual(backtest_config.commission, 0.001)
        
    def test_exchange_config(self):
        """Test exchange configuration."""
        exchange_config = ExchangeConfig(
            name="bybit",
            testnet=True,
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://api-testnet.bybit.com",
            timeout=10,
            max_retries=3,
            retry_delay=5,
            rate_limit={"calls": 2, "period": 1}
        )
        self.assertEqual(exchange_config.name, "bybit")
        self.assertTrue(exchange_config.testnet)
        
    def test_logging_config(self):
        """Test logging configuration."""
        logging_config = LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file="logs/test.log"
        )
        self.assertEqual(logging_config.level, "INFO")
        self.assertEqual(logging_config.max_size, 10485760)  # Default value
        
    def test_bot_config(self):
        """Test main bot configuration."""
        # Create sub-configs
        symbol_config = SymbolConfig(
            min_volume=1000000,
            min_volatility=0.001,
            max_spread=0.001,
            leverage=1
        )
        
        risk_config = RiskConfig(
            stop_loss={"type": "fixed", "value": 0.02},
            take_profit={"type": "fixed", "value": 0.04},
            trailing_stop={"enabled": True, "activation": 0.01, "distance": 0.005},
            max_daily_loss=0.02,
            max_drawdown=0.15,
            min_risk_reward=2.0,
            base_risk_per_trade=0.01,
            max_open_trades=3,
            max_daily_trades=10
        )
        
        backtest_config = BacktestConfig(
            initial_capital=10000,
            commission=0.001,
            slippage=0.001
        )
        
        exchange_config = ExchangeConfig(
            name="bybit",
            testnet=True,
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://api-testnet.bybit.com",
            timeout=10,
            max_retries=3,
            retry_delay=5,
            rate_limit={"calls": 2, "period": 1}
        )
        
        logging_config = LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file="logs/test.log"
        )
        
        # Create main config
        bot_config = BotConfig(
            symbols={"BTCUSDT": symbol_config},
            timeframes=["1m", "5m", "15m"],
            update_interval=1,
            polling_interval=1,
            report_interval=3600,
            risk=risk_config,
            backtest=backtest_config,
            exchange=exchange_config,
            logging=logging_config
        )
        
        self.assertEqual(bot_config.timeframes, ["1m", "5m", "15m"])
        self.assertEqual(bot_config.update_interval, 1)
        self.assertEqual(bot_config.risk.max_daily_loss, 0.02)
        self.assertEqual(bot_config.exchange.name, "bybit")

if __name__ == "__main__":
    unittest.main() 