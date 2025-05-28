import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os

class PerformanceAnalyzer:
    def __init__(self, initial_capital: float):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        
        # Configure pandas for better performance
        pd.options.mode.chained_assignment = None
        
        # Cache for calculated metrics
        self._metrics_cache = {}
    
    @lru_cache(maxsize=100)
    def _calculate_basic_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate basic trading metrics"""
        total_trades = len(trades_df[trades_df['type'] == 'exit'])
        winning_trades = len(trades_df[(trades_df['type'] == 'exit') & (trades_df['pnl'] > 0)])
        losing_trades = len(trades_df[(trades_df['type'] == 'exit') & (trades_df['pnl'] <= 0)])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0
        }
    
    @lru_cache(maxsize=100)
    def _calculate_profit_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate profit-related metrics"""
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        
        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf')
        }
    
    @lru_cache(maxsize=100)
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate risk metrics"""
        returns = trades_df[trades_df['type'] == 'exit']['pnl']
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate drawdown
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Calculate risk-adjusted returns
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        }
    
    @lru_cache(maxsize=100)
    def _calculate_trade_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trade statistics"""
        returns = trades_df[trades_df['type'] == 'exit']['pnl']
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in returns:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            'avg_trade': returns.mean() if not returns.empty else 0,
            'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
            'avg_loss': returns[returns <= 0].mean() if len(returns[returns <= 0]) > 0 else 0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyze trading results with parallel processing"""
        try:
            if not trades:
                return self._get_empty_metrics()
            
            # Convert trades to DataFrame with optimized settings
            trades_df = pd.DataFrame(trades)
            trades_df['pnl'] = trades_df['pnl'].astype('float32')
            
            # Calculate metrics in parallel
            with ProcessPoolExecutor() as executor:
                futures = {
                    'basic': executor.submit(self._calculate_basic_metrics, trades_df),
                    'profit': executor.submit(self._calculate_profit_metrics, trades_df),
                    'risk': executor.submit(self._calculate_risk_metrics, trades_df),
                    'stats': executor.submit(self._calculate_trade_stats, trades_df)
                }
                
                results = {}
                for category, future in futures.items():
                    results.update(future.result())
            
            # Calculate additional metrics
            results['total_return'] = (1 + trades_df[trades_df['type'] == 'exit']['pnl']).prod() - 1
            
            # Cache results
            self._metrics_cache[str(trades_df.hash())] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {str(e)}", exc_info=True)
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'volatility': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self._metrics_cache.clear()
        self._calculate_basic_metrics.cache_clear()
        self._calculate_profit_metrics.cache_clear()
        self._calculate_risk_metrics.cache_clear()
        self._calculate_trade_stats.cache_clear() 