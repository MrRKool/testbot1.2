import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from bot.backtesting.backtest_engine import BacktestEngine
from bot.strategies.moving_average_cross import MovingAverageCrossStrategy

def generate_sample_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate sample price data for testing."""
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Generate random walk price
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, len(dates))
    price = 50000 * (1 + returns).cumprod()
    
    # Generate volume
    volume = np.random.lognormal(10, 1, len(dates))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': price * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': price,
        'volume': volume
    }, index=dates)
    
    return data

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Generate sample data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        market_data = {
            'BTCUSDT': generate_sample_data(start_date, end_date)
        }
        
        # Initialize strategy and backtest engine
        strategy = MovingAverageCrossStrategy()
        backtest_engine = BacktestEngine()
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtest_engine.run_backtest(
            strategy=strategy,
            market_data=market_data,
            start_date=start_date,
            end_date=end_date
        )
        
        if results:
            # Print results
            logger.info("\nBacktest Results:")
            logger.info(f"Total Trades: {results.total_trades}")
            logger.info(f"Winning Trades: {results.winning_trades}")
            logger.info(f"Losing Trades: {results.losing_trades}")
            logger.info(f"Win Rate: {results.win_rate:.2%}")
            logger.info(f"Profit Factor: {results.profit_factor:.2f}")
            logger.info(f"Total Profit: ${results.total_profit:.2f}")
            logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
            logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            
            # Plot equity curve
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            results.equity_curve.plot()
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.show()
            
        else:
            logger.error("Backtest failed")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 