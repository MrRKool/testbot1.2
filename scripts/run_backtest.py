#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.backtest_engine import BacktestEngine
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.config.config_loader import ConfigLoader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backtest.log')
        ]
    )

def analyze_trades(trades: pd.DataFrame) -> dict:
    """Analyze trade statistics"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration': 0,
            'avg_daily_trades': 0
        }
    
    # Calculate basic metrics
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate returns and drawdown
    cumulative_returns = trades['pnl'].cumsum()
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
    
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
    
    # Calculate risk-adjusted returns
    returns = trades['pnl'].pct_change()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
    
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
    
    # Calculate trade statistics
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
    largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
    
    # Calculate consecutive wins/losses
    trades['win'] = trades['pnl'] > 0
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for win in trades['win']:
        if win:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    # Calculate trade duration
    trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600  # in hours
    avg_trade_duration = trades['duration'].mean() if len(trades) > 0 else 0
    
    # Calculate daily statistics
    trades['date'] = trades['entry_time'].dt.date
    daily_trades = trades.groupby('date').size()
    avg_daily_trades = daily_trades.mean() if len(daily_trades) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'total_return': total_return * 100,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_trade_duration': avg_trade_duration,
        'avg_daily_trades': avg_daily_trades
    }

def main():
    """Main function to run backtest"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_loader = ConfigLoader('config/config.yaml')
        config = config_loader.load_config()
        
        # Initialize backtest engine
        engine = BacktestEngine(config)
        
        # Set parameters for backtest
        symbol = "BTCUSDT"  # Change as needed
        timeframe = "1h"    # Change to "1m" or "5m" as needed
        start_date = "2024-01-01"
        end_date = "2024-07-17"
        
        logger.info(f"Starting backtest for {symbol} {timeframe} from {start_date} to {end_date}")
        
        results = engine.run_backtest(symbol, timeframe, start_date, end_date)
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(results['trades'], columns=[
            'entry_time', 'exit_time', 'entry_price', 'exit_price', 'size', 'pnl'
        ])
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Analyze results
        stats = analyze_trades(trades_df)
        
        # Log results
        logger.info("\nBacktest Results:")
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
        logger.info(f"Total Return: {stats['total_return']:.2f}%")
        logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"Sortino Ratio: {stats['sortino_ratio']:.2f}")
        
        logger.info("\nTrade Statistics:")
        logger.info(f"Average Win: {stats['avg_win']:.2f}")
        logger.info(f"Average Loss: {stats['avg_loss']:.2f}")
        logger.info(f"Largest Win: {stats['largest_win']:.2f}")
        logger.info(f"Largest Loss: {stats['largest_loss']:.2f}")
        logger.info(f"Max Consecutive Wins: {stats['max_consecutive_wins']}")
        logger.info(f"Max Consecutive Losses: {stats['max_consecutive_losses']}")
        logger.info(f"Average Trade Duration: {stats['avg_trade_duration']:.2f} hours")
        logger.info(f"Average Daily Trades: {stats['avg_daily_trades']:.2f}")
        
        # Save results to CSV
        trades_df.to_csv(f'backtest_results_{symbol}.csv', index=False)
        logger.info(f"\nDetailed results saved to backtest_results_{symbol}.csv")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 