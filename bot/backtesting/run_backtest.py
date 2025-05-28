import logging
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from bot.backtesting.backtest_engine import BacktestEngine
from bot.strategies.example_strategy import ExampleStrategy

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load backtest configuration."""
    config = {
        # Backtest parameters
        'initial_capital': 10000,
        'position_size': 0.1,  # 10% of capital
        'max_positions': 5,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005,  # 0.05%
        
        # Risk management
        'stop_loss': 0.02,  # 2%
        'take_profit': 0.04,  # 4%
        'max_drawdown': 0.2,  # 20%
        
        # Strategy parameters
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'sma_fast': 20,
        'sma_slow': 50,
        
        # Results directory
        'results_dir': 'results/backtest'
    }
    return config

def load_local_data(data_path: str) -> pd.DataFrame:
    """Load local data for backtesting."""
    try:
        # Load data from local file
        data = pd.read_csv(data_path)
        
        # Convert timestamp to datetime index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Rename columns to match expected format
        data.columns = [col.lower() for col in data.columns]
        
        return data
        
    except Exception as e:
        print(f"Error loading local data: {str(e)}")
        return pd.DataFrame()

def main():
    """Run backtest."""
    # Setup logging
    log = setup_logging()
    log.info("Starting backtest...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create results directory
        results_dir = Path(config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        # Load local data
        data_path = 'data/BTCUSDT/btcusdt_data_1h.csv'  # Using 1h timeframe data
        log.info(f"Loading data from {data_path}...")
        data = load_local_data(data_path)
        
        if data.empty:
            log.error("No data loaded")
            return
            
        log.info(f"Loaded {len(data)} data points")
        
        # Initialize strategy
        strategy = ExampleStrategy(config)
        
        # Initialize backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest
        log.info("Running backtest...")
        results = engine.run_backtest(data, strategy)
        
        # Print results
        log.info("\nBacktest Results:")
        log.info(f"Total Trades: {results['metrics']['total_trades']}")
        log.info(f"Win Rate: {results['metrics']['win_rate']:.2%}")
        log.info(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
        log.info(f"Total Return: {results['metrics']['total_return']:.2%}")
        log.info(f"Annualized Return: {results['metrics']['annualized_return']:.2%}")
        log.info(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
        log.info(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        log.info(f"Sortino Ratio: {results['metrics']['sortino_ratio']:.2f}")
        
        log.info("\nResults saved to: " + str(results_dir))
        
    except Exception as e:
        log.error(f"Error in backtest: {str(e)}")
        raise

if __name__ == "__main__":
    main() 