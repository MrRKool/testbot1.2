import logging
import json
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from bot.backtesting.optimization_engine import OptimizationEngine

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/optimization.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load optimization configuration."""
    config = {
        # Optimization parameters
        'results_dir': 'results/optimization',
        
        # Performance thresholds
        'min_sharpe': 1.0,
        'min_profit_factor': 1.5,
        'min_win_rate': 0.45,
        
        # Trading parameters
        'initial_capital': 10000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005,  # 0.05%
    }
    return config

def main():
    """Run optimization."""
    # Setup logging
    log = setup_logging()
    log.info("Starting optimization...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create results directory
        results_dir = Path(config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        # Initialize optimization engine
        engine = OptimizationEngine(config)
        
        # Run continuous optimization
        engine.run_continuous_optimization(symbol='BTCUSDT')
        
    except Exception as e:
        log.error(f"Error in optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 