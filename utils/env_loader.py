import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv('testbot1.env')

# API Configuration
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = {
        "API_KEY": API_KEY,
        "API_SECRET": API_SECRET,
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logging.warning("⚠️ Warning: The following variables are not set:")
        for var in missing_vars:
            logging.warning(f"  - {var}")
        return False
    
    logging.info("✅ All required variables are correctly set")
    return True

def get_api_keys():
    """Get API keys from environment variables."""
    return API_KEY, API_SECRET

def get_telegram_config():
    """Get Telegram configuration from environment variables."""
    return TELEGRAM_TOKEN, TELEGRAM_CHAT_ID 