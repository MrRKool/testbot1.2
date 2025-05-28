import os
import sys
import subprocess
import logging
from pathlib import Path
import json

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup.log')
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories."""
    directories = [
        'logs',
        'data',
        'config',
        'models',
        'backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_virtual_environment():
    """Create and setup virtual environment."""
    try:
        # Create virtual environment
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        logger.info("Created virtual environment")
        
        # Determine pip path
        if sys.platform == 'win32':
            pip_path = 'venv\\Scripts\\pip'
            python_path = 'venv\\Scripts\\python'
        else:
            pip_path = 'venv/bin/pip'
            python_path = 'venv/bin/python'
            
        # Upgrade pip
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        logger.info("Upgraded pip")
        
        # Install requirements
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        logger.info("Installed requirements")
        
        return python_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up virtual environment: {str(e)}")
        sys.exit(1)

def create_config_files():
    """Create default configuration files."""
    config_files = {
        'config/bot_config.json': {
            'trading': {
                'exchange': 'binance',
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframe': '1h',
                'max_positions': 3,
                'risk_per_trade': 0.02
            },
            'monitoring': {
                'check_interval': 300,
                'alert_thresholds': {
                    'cpu_percent': 80,
                    'memory_percent': 80,
                    'disk_percent': 80,
                    'api_latency_ms': 1000
                }
            }
        },
        'config/voice_config.json': {
            'voice_id': 'dutch',
            'rate': 150,
            'volume': 1.0,
            'commands': {
                'start_trading': ['start trading', 'begin trading', 'start bot'],
                'stop_trading': ['stop trading', 'stop bot', 'halt trading'],
                'check_status': ['status', 'how are you', 'what\'s your status'],
                'show_metrics': ['show metrics', 'show performance', 'show results'],
                'adjust_risk': ['adjust risk', 'change risk', 'modify risk'],
                'change_strategy': ['change strategy', 'switch strategy', 'new strategy']
            }
        }
    }
    
    for file_path, config in config_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Created config file: {file_path}")

def setup_environment_variables():
    """Setup environment variables."""
    env_vars = {
        'BOT_ENV': 'development',
        'LOG_LEVEL': 'INFO',
        'API_KEY': '',
        'API_SECRET': '',
        'TELEGRAM_TOKEN': '',
        'TELEGRAM_CHAT_ID': ''
    }
    
    env_file = '.env'
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        logger.info("Created .env file")
        logger.warning("Please update the .env file with your API keys and tokens")

def main():
    """Main setup function."""
    global logger
    logger = setup_logging()
    
    logger.info("Starting bot setup...")
    
    # Create directories
    create_directories()
    
    # Setup virtual environment
    python_path = setup_virtual_environment()
    
    # Create config files
    create_config_files()
    
    # Setup environment variables
    setup_environment_variables()
    
    logger.info("Setup completed successfully!")
    logger.info(f"To start the bot, activate the virtual environment and run: {python_path} main.py")

if __name__ == "__main__":
    main() 