import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from logging.handlers import RotatingFileHandler

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_logging():
    """Setup logging configuration with rotation"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                'logs/trade_summary.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )

def load_trade_data(file_path: Path) -> pd.DataFrame:
    """Load trade data from CSV file with optimized settings"""
    try:
        # Try to load memory map first
        mmap_path = file_path.with_suffix('.mmap')
        if mmap_path.exists():
            return pd.read_pickle(mmap_path)
        
        # Fall back to CSV if memory map doesn't exist
        df = pd.read_csv(
            file_path,
            parse_dates=['entry_time', 'exit_time'],
            dtype={
                'symbol': 'category',
                'entry_price': 'float32',
                'exit_price': 'float32',
                'size': 'float32',
                'pnl': 'float32',
                'fees': 'float32',
                'slippage': 'float32'
            }
        )
        
        # Create memory map for future use
        df.to_pickle(mmap_path, protocol=4)
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading trade data from {file_path}: {str(e)}")
        return pd.DataFrame()

def calculate_basic_metrics(df: pd.DataFrame) -> Dict:
    """Calculate basic trading metrics"""
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0
    }

def calculate_pnl_metrics(df: pd.DataFrame) -> Dict:
    """Calculate PnL related metrics"""
    total_pnl = df['pnl'].sum()
    total_fees = df['fees'].sum()
    total_slippage = df['slippage'].sum()
    
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0
    
    return {
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    }

def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
    """Calculate risk metrics"""
    # Calculate returns
    df['return'] = df['pnl'] / (df['entry_price'] * df['size'])
    
    # Calculate drawdown
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['rolling_max'] = df['cumulative_pnl'].expanding().max()
    df['drawdown'] = (df['cumulative_pnl'] - df['rolling_max']) / df['rolling_max']
    
    # Calculate ratios
    returns = df['return'].mean() * 252  # Annualized return
    volatility = df['return'].std() * np.sqrt(252)  # Annualized volatility
    downside_returns = df[df['return'] < 0]['return']
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    return {
        'max_drawdown': abs(df['drawdown'].min()),
        'sharpe_ratio': (returns - 0.02) / volatility if volatility != 0 else 0,
        'sortino_ratio': (returns - 0.02) / downside_std if downside_std != 0 else 0,
        'volatility': volatility
    }

def calculate_time_metrics(df: pd.DataFrame) -> Dict:
    """Calculate time-based metrics"""
    df['holding_time'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    
    return {
        'avg_holding_time': df['holding_time'].mean(),
        'trades_per_day': len(df) / ((df['exit_time'].max() - df['entry_time'].min()).days + 1)
    }

def calculate_metrics_parallel(df: pd.DataFrame) -> Dict:
    """Calculate all metrics in parallel"""
    with ProcessPoolExecutor() as executor:
        futures = {
            'basic': executor.submit(calculate_basic_metrics, df),
            'pnl': executor.submit(calculate_pnl_metrics, df),
            'risk': executor.submit(calculate_risk_metrics, df),
            'time': executor.submit(calculate_time_metrics, df)
        }
        
        results = {}
        for category, future in futures.items():
            results.update(future.result())
        
        return results

@lru_cache(maxsize=100)
def calculate_rolling_metrics(df_hash: str, window: int) -> Dict:
    """Calculate rolling metrics for a given window"""
    # This function would be implemented to calculate rolling metrics
    # The df_hash parameter is used for caching
    pass

def plot_cumulative_pnl(df: pd.DataFrame, output_dir: Path):
    """Plot cumulative PnL with enhanced styling"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['exit_time'], df['pnl'].cumsum(), linewidth=2)
    plt.title('Cumulative PnL', fontsize=14, pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Cumulative PnL', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_pnl.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pnl_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot PnL distribution with enhanced styling"""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['pnl'], bins=50, kde=True)
    plt.title('Trade PnL Distribution', fontsize=14, pad=20)
    plt.xlabel('PnL', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pnl_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_win_rate_by_hour(df: pd.DataFrame, output_dir: Path):
    """Plot win rate by hour with enhanced styling"""
    plt.figure(figsize=(12, 6))
    df['hour'] = df['entry_time'].dt.hour
    win_rate_by_hour = df.groupby('hour')['pnl'].apply(lambda x: (x > 0).mean())
    win_rate_by_hour.plot(kind='bar', color='skyblue')
    plt.title('Win Rate by Hour', fontsize=14, pad=20)
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_holding_time(df: pd.DataFrame, output_dir: Path):
    """Plot holding time distribution with enhanced styling"""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['holding_time'], bins=50, kde=True)
    plt.title('Trade Holding Time Distribution', fontsize=14, pad=20)
    plt.xlabel('Holding Time (hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'holding_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_plots_parallel(df: pd.DataFrame, output_dir: Path):
    """Generate all plots in parallel"""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(plot_cumulative_pnl, df, output_dir),
            executor.submit(plot_pnl_distribution, df, output_dir),
            executor.submit(plot_win_rate_by_hour, df, output_dir),
            executor.submit(plot_holding_time, df, output_dir)
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error generating plot: {str(e)}")

def summarize_trades():
    """Main function to summarize trades"""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting trade summary")
        
        # Get results directory
        results_dir = Path('results')
        if not results_dir.exists():
            logging.error("Results directory not found")
            return
        
        # Find all trade files
        trade_files = list(results_dir.glob('*_trades_*.csv'))
        if not trade_files:
            logging.error("No trade files found")
            return
        
        # Process each trade file
        for file_path in trade_files:
            logging.info(f"Processing {file_path}")
            
            # Load trade data
            df = load_trade_data(file_path)
            if df.empty:
                continue
            
            # Calculate metrics in parallel
            metrics = calculate_metrics_parallel(df)
            
            # Save metrics
            metrics_file = file_path.parent / f"{file_path.stem}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Generate plots in parallel
            generate_plots_parallel(df, file_path.parent)
            
            # Log summary
            logging.info(f"\nSummary for {file_path.name}:")
            logging.info(f"Total Trades: {metrics['total_trades']}")
            logging.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logging.info(f"Total PnL: {metrics['total_pnl']:.2f}")
            logging.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logging.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logging.info(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        
        logging.info("Trade summary completed")
        
    except Exception as e:
        logging.error(f"Error in trade summary: {str(e)}")

if __name__ == "__main__":
    summarize_trades() 