import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bot.backtesting.backtest_engine import BacktestEngine
from src.ai.ai_strategy import AITradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(days=30):
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='H')
    
    # Generate random price data with some trend and volatility
    np.random.seed(42)
    base_price = 50000
    trend = np.linspace(0, 5000, len(dates))  # Upward trend
    noise = np.random.normal(0, 500, len(dates))  # Random noise
    prices = base_price + trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    # Calculate some basic indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def load_config():
    """Load configuration from yaml file."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def main():
    """Run backtest with AI strategy."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Generate sample data
        data = generate_sample_data(days=30)
        logger.info(f"Generated {len(data)} data points")

        # Zet index op timestamp en verwijder dubbele timestamps
        data = data.set_index('timestamp')
        data = data[~data.index.duplicated(keep='first')]
        
        # Initialize AI strategy
        strategy = AITradingStrategy(config)
        logger.info("Initialized AI strategy")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.001,
            risk_free_rate=0.02
        )
        logger.info("Initialized backtest engine")
        
        # Prepare market data
        market_data = {'BTCUSDT': data}
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Run backtest
        results = engine.run_backtest(
            strategy=strategy,
            market_data=market_data,
            start_date=start_date,
            end_date=end_date
        )
        
        if results:
            # Log results
            logger.info(f"Total trades: {results.total_trades}")
            logger.info(f"Win rate: {results.win_rate:.2%}")
            logger.info(f"Profit factor: {results.profit_factor:.2f}")
            logger.info(f"Total profit: ${results.total_profit:.2f}")
            logger.info(f"Max drawdown: {results.max_drawdown:.2%}")
            logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            results.equity_curve.plot()
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.savefig('equity_curve.png')
            logger.info("Saved equity curve plot")
        else:
            logger.error("Backtest failed")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 