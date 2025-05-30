import os
import sys
import yaml
import logging
from datetime import datetime, timedelta

# Voeg src toe aan sys.path zodat imports werken
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.historical_data import HistoricalDataFetcher
from src.training.strategy_trainer import StrategyTrainer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load training configuration."""
    config_path = os.path.join('config', 'training_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(params, performance, symbol, timeframe):
    """Save training results to file."""
    results_dir = os.path.join('results', 'training')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"
    
    results = {
        'parameters': vars(params),
        'performance': performance,
        'symbol': symbol,
        'timeframe': timeframe,
        'timestamp': timestamp
    }
    
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """Main training script."""
    # Load configuration
    config = load_config()
    
    # Initialize exchange
    exchange_config = {
        'apiKey': os.getenv('EXCHANGE_API_KEY'),
        'secret': os.getenv('EXCHANGE_SECRET'),
        'enableRateLimit': True
    }
    
    # Process each symbol and timeframe
    for symbol in config['historical_data']['symbols']:
        for timeframe in config['historical_data']['timeframes']:
            logger.info(f"Processing {symbol} on {timeframe} timeframe")
            
            try:
                # Fetch historical data
                fetcher = HistoricalDataFetcher('binance', exchange_config)
                since = datetime.now() - timedelta(days=config['historical_data']['lookback_days'])
                
                historical_data = fetcher.fetch_multiple_timeframes(
                    symbol,
                    [timeframe],
                    since=since
                )
                
                # Initialize and run strategy trainer
                trainer = StrategyTrainer(
                    historical_data,
                    initial_balance=config['optimization']['initial_balance']
                )
                
                # Optimize strategy
                best_params, best_performance = trainer.optimize_strategy(
                    n_trials=config['optimization']['n_trials'],
                    timeframe=timeframe
                )
                
                # Save results
                save_results(best_params, best_performance, symbol, timeframe)
                
                logger.info(f"Completed training for {symbol} on {timeframe}")
                logger.info(f"Best Sharpe Ratio: {best_performance['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {best_performance['max_drawdown']:.2%}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} on {timeframe}: {str(e)}")
                continue

if __name__ == "__main__":
    main() 