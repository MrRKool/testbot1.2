import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os

class PerformanceAnalyzer:
    def __init__(self, config: Dict):
        """Initialize performance analyzer with optimized settings."""
        self.config = config
        self.log = logging.getLogger(__name__)
        
        # Initialize metrics
        self._metrics = {
            'trades': [],
            'daily_pnl': [],
            'drawdown': [],
            'win_rate': [],
            'risk_reward': [],
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Initialize cache
        self._cache = {}
        
    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyze trades with parallel processing."""
        try:
            # Process trades in parallel
            with ProcessPoolExecutor() as executor:
                # Calculate basic metrics
                total_trades = len(trades)
                winning_trades = sum(1 for t in trades if t['pnl'] > 0)
                losing_trades = sum(1 for t in trades if t['pnl'] <= 0)
                
                # Calculate PnL metrics
                pnl_future = executor.submit(self._calculate_pnl_metrics, trades)
                
                # Calculate risk metrics
                risk_future = executor.submit(self._calculate_risk_metrics, trades)
                
                # Calculate performance ratios
                ratios_future = executor.submit(self._calculate_performance_ratios, trades)
                
                # Get results
                pnl_metrics = pnl_future.result()
                risk_metrics = risk_future.result()
                performance_ratios = ratios_future.result()
                
            # Combine all metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                **pnl_metrics,
                **risk_metrics,
                **performance_ratios
            }
            
            # Update internal metrics
            self._update_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.log.error(f"Error analyzing trades: {str(e)}")
            return {}
            
    def _calculate_pnl_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate PnL-related metrics."""
        try:
            # Calculate daily PnL
            daily_pnl = {}
            for trade in trades:
                date = trade['timestamp'].date()
                if date not in daily_pnl:
                    daily_pnl[date] = 0
                daily_pnl[date] += trade['pnl']
                
            # Calculate PnL metrics
            total_pnl = sum(trade['pnl'] for trade in trades)
            avg_pnl = total_pnl / len(trades) if trades else 0
            max_pnl = max(trade['pnl'] for trade in trades) if trades else 0
            min_pnl = min(trade['pnl'] for trade in trades) if trades else 0
            
            return {
                'total_pnl': total_pnl,
                'average_pnl': avg_pnl,
                'max_pnl': max_pnl,
                'min_pnl': min_pnl,
                'daily_pnl': daily_pnl
            }
            
        except Exception as e:
            self.log.error(f"Error calculating PnL metrics: {str(e)}")
            return {}
            
    def _calculate_risk_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate risk-related metrics."""
        try:
            # Calculate drawdown
            cumulative_pnl = 0
            peak = 0
            drawdown = 0
            max_drawdown = 0
            
            for trade in trades:
                cumulative_pnl += trade['pnl']
                peak = max(peak, cumulative_pnl)
                drawdown = peak - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)
                
            # Calculate volatility
            returns = [trade['pnl'] for trade in trades]
            volatility = np.std(returns) if returns else 0
            
            return {
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'drawdown': drawdown
            }
            
        except Exception as e:
            self.log.error(f"Error calculating risk metrics: {str(e)}")
            return {}
            
    def _calculate_performance_ratios(self, trades: List[Dict]) -> Dict:
        """Calculate performance ratios."""
        try:
            # Calculate returns
            returns = [trade['pnl'] for trade in trades]
            if not returns:
                return {
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'risk_reward_ratio': 0
                }
                
            # Calculate Sharpe ratio
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            sortino_ratio = avg_return / downside_std if downside_std > 0 else 0
            
            # Calculate risk-reward ratio
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'risk_reward_ratio': risk_reward
            }
            
        except Exception as e:
            self.log.error(f"Error calculating performance ratios: {str(e)}")
            return {}
            
    def _update_metrics(self, metrics: Dict):
        """Update internal metrics."""
        try:
            self._metrics['trades'].extend(metrics.get('trades', []))
            self._metrics['daily_pnl'].append(metrics.get('daily_pnl', {}))
            self._metrics['drawdown'].append(metrics.get('drawdown', 0))
            self._metrics['win_rate'].append(metrics.get('win_rate', 0))
            self._metrics['risk_reward'].append(metrics.get('risk_reward_ratio', 0))
            self._metrics['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
            self._metrics['sortino_ratio'].append(metrics.get('sortino_ratio', 0))
            self._metrics['max_drawdown'] = max(self._metrics['max_drawdown'], metrics.get('max_drawdown', 0))
            self._metrics['total_trades'] += metrics.get('total_trades', 0)
            self._metrics['winning_trades'] += metrics.get('winning_trades', 0)
            self._metrics['losing_trades'] += metrics.get('losing_trades', 0)
            
        except Exception as e:
            self.log.error(f"Error updating metrics: {str(e)}")
            
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        try:
            return {
                'total_trades': self._metrics['total_trades'],
                'winning_trades': self._metrics['winning_trades'],
                'losing_trades': self._metrics['losing_trades'],
                'win_rate': self._metrics['win_rate'][-1] if self._metrics['win_rate'] else 0,
                'max_drawdown': self._metrics['max_drawdown'],
                'sharpe_ratio': self._metrics['sharpe_ratio'][-1] if self._metrics['sharpe_ratio'] else 0,
                'sortino_ratio': self._metrics['sortino_ratio'][-1] if self._metrics['sortino_ratio'] else 0,
                'risk_reward_ratio': self._metrics['risk_reward'][-1] if self._metrics['risk_reward'] else 0
            }
        except Exception as e:
            self.log.error(f"Error getting metrics: {str(e)}")
            return {}
            
    def generate_report(self) -> str:
        """Generate performance report."""
        try:
            metrics = self.get_metrics()
            
            report = f"""
Performance Report
=================

Trading Statistics:
------------------
Total Trades: {metrics['total_trades']}
Winning Trades: {metrics['winning_trades']}
Losing Trades: {metrics['losing_trades']}
Win Rate: {metrics['win_rate']:.2%}

Performance Metrics:
------------------
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2%}
"""
            return report
            
        except Exception as e:
            self.log.error(f"Error generating report: {str(e)}")
            return "Error generating performance report"

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
            self._cache[str(trades_df.hash())] = results
            
            return results
            
        except Exception as e:
            self.log.error(f"Error analyzing trades: {str(e)}", exc_info=True)
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
        self._cache.clear()
        self._calculate_basic_metrics.cache_clear()
        self._calculate_profit_metrics.cache_clear()
        self._calculate_risk_metrics.cache_clear()
        self._calculate_trade_stats.cache_clear() 