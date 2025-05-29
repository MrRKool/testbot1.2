import os
import logging
import unittest
from datetime import datetime
from bot.config import load_config, save_config, validate_config, DEFAULTS, ConfigManager
from utils.env_loader import check_environment, get_api_keys, get_telegram_config

class TestBot(unittest.TestCase):
    """Test cases for the trading bot."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Create test config
        cls.test_config = DEFAULTS.copy()
        cls.test_config_path = "config/test_config.yaml"
        
    def setUp(self):
        """Set up each test case."""
        self.config_manager = ConfigManager()
        
    def test_environment_variables(self):
        """Test environment variables."""
        self.logger.info("Testing environment variables...")
        
        # Check environment variables
        self.assertTrue(check_environment(), "Environment variables check failed")
        
        # Test API keys
        api_key, api_secret = get_api_keys()
        self.assertIsNotNone(api_key, "API key is None")
        self.assertIsNotNone(api_secret, "API secret is None")
        
        # Test Telegram config
        telegram_token, telegram_chat_id = get_telegram_config()
        self.assertIsNotNone(telegram_token, "Telegram token is None")
        self.assertIsNotNone(telegram_chat_id, "Telegram chat ID is None")
        
    def test_config_loading(self):
        """Test configuration loading."""
        self.logger.info("Testing configuration loading...")
        
        # Test loading default config
        config = load_config()
        self.assertIsNotNone(config, "Failed to load default configuration")
        self.assertTrue(validate_config(config), "Default configuration validation failed")
        
        # Test saving and loading custom config
        self.logger.info("Testing custom configuration...")
        custom_config = self.test_config.copy()
        custom_config["trading"]["symbols"]["BTCUSDT"]["leverage"] = 2
        custom_config["risk"]["max_daily_loss"] = 0.03
        
        # Save custom config
        save_config(custom_config, self.test_config_path)
        
        # Load custom config
        loaded_config = load_config(self.test_config_path)
        self.assertIsNotNone(loaded_config, "Failed to load custom configuration")
        self.assertTrue(validate_config(loaded_config), "Custom configuration validation failed")
        self.assertEqual(
            loaded_config["trading"]["symbols"]["BTCUSDT"]["leverage"],
            2,
            "Custom config not loaded correctly"
        )
        
    def test_config_validation(self):
        """Test configuration validation."""
        self.logger.info("Testing configuration validation...")
        
        # Test valid config
        self.assertTrue(validate_config(self.test_config), "Valid configuration failed validation")
        
        # Test invalid config
        invalid_config = self.test_config.copy()
        del invalid_config["trading"]["symbols"]["BTCUSDT"]["leverage"]
        self.assertFalse(validate_config(invalid_config), "Invalid configuration passed validation")
        
    def test_config_manager(self):
        """Test configuration manager."""
        self.logger.info("Testing configuration manager...")
        
        # Test caching
        config1 = self.config_manager.load_config()
        config2 = self.config_manager.load_config()
        self.assertEqual(id(config1), id(config2), "Configuration caching failed")
        
        # Test cache invalidation
        self.config_manager.clear_cache()
        config3 = self.config_manager.load_config()
        self.assertNotEqual(id(config1), id(config3), "Cache invalidation failed")
        
    def test_trading_config(self):
        """Test trading configuration."""
        self.logger.info("Testing trading configuration...")
        config = load_config()
        
        # Test symbols
        self.assertIn("BTCUSDT", config["trading"]["symbols"], "BTCUSDT not found in symbols")
        self.assertIn("ETHUSDT", config["trading"]["symbols"], "ETHUSDT not found in symbols")
        
        # Test timeframes
        self.assertIn("1h", config["trading"]["timeframes"], "1h timeframe not found")
        
        # Test risk parameters
        self.assertGreater(config["risk"]["max_daily_loss"], 0, "Invalid max_daily_loss")
        self.assertGreater(config["risk"]["max_drawdown"], 0, "Invalid max_drawdown")
        self.assertGreater(config["risk"]["min_risk_reward"], 1, "Invalid min_risk_reward")
        
    def test_exchange_config(self):
        """Test exchange configuration."""
        self.logger.info("Testing exchange configuration...")
        config = load_config()
        
        # Test exchange settings
        self.assertEqual(config["exchange"]["name"], "bybit", "Invalid exchange name")
        self.assertIsInstance(config["exchange"]["testnet"], bool, "Invalid testnet value")
        self.assertIn("rate_limit", config["exchange"], "Missing rate_limit")
        
    def test_logging_config(self):
        """Test logging configuration."""
        self.logger.info("Testing logging configuration...")
        config = load_config()
        
        # Test logging settings
        self.assertIn(config["logging"]["level"], 
                     ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                     "Invalid logging level")
        self.assertGreater(config["logging"]["max_size"], 0, "Invalid max_size")
        self.assertGreater(config["logging"]["backup_count"], 0, "Invalid backup_count")
        
    def tearDown(self):
        """Clean up after each test case."""
        # Remove test config file if it exists
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test config file if it exists
        if os.path.exists(cls.test_config_path):
            os.remove(cls.test_config_path)

if __name__ == "__main__":
    unittest.main(verbosity=2) 