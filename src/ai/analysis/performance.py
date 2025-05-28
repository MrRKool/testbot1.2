import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

class PerformanceAnalyzer:
    """Analyzes trading performance and generates metrics."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance metrics
        self.trades = []
        self.daily_returns = pd.Series()
        self.equity_curve = pd.Series()
        
        # Create cache directory
        self.cache_dir = "cache/performance"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load historical performance data
        self._load_performance_data()
        
    def _load_performance_data(self):
        """Load historical performance data from cache."""
        try:
            # Load trades
            trades_file = os.path.join(self.cache_dir, "trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self.trades = json.load(f)
                    
            # Load daily returns
            returns_file = os.path.join(self.cache_dir, "daily_returns.csv")
            if os.path.exists(returns_file):
                self.daily_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
                
            # Load equity curve
            equity_file = os.path.join(self.cache_dir, "equity_curve.csv")
            if os.path.exists(equity_file):
                self.equity_curve = pd.read_csv(equity_file, index_col=0, parse_dates=True)
                
        except Exception as e:
            self.logger.error(f"Error loading performance data: {str(e)}")
            
    def _save_performance_data(self):
        """Save current performance data to cache."""
        try:
            # Save trades
            trades_file = os.path.join(self.cache_dir, "trades.json")
            with open(trades_file, 'w') as f:
                json.dump(self.trades, f)
                
            # Save daily returns
            returns_file = os.path.join(self.cache_dir, "daily_returns.csv")
            self.daily_returns.to_csv(returns_file)
            
            # Save equity curve
            equity_file = os.path.join(self.cache_dir, "equity_curve.csv")
            self.equity_curve.to_csv(equity_file)
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")
            
    def add_trade(self, trade: Dict):
        """Add a new trade to the performance analysis."""
        try:
            # Add trade to list
            self.trades.append(trade)
            
            # Update daily returns
            self._update_daily_returns(trade)
            
            # Update equity curve
            self._update_equity_curve(trade)
            
            # Save updated data
            self._save_performance_data()
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {str(e)}")
            
    def _update_daily_returns(self, trade: Dict):
        """Update daily returns with new trade."""
        try:
            # Get trade date and P&L
            date = pd.to_datetime(trade['exit_time'])
            pnl = trade['pnl']
            
            # Add to daily returns
            if date in self.daily_returns.index:
                self.daily_returns[date] += pnl
            else:
                self.daily_returns[date] = pnl
                
            # Sort index
            self.daily_returns = self.daily_returns.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error updating daily returns: {str(e)}")
            
    def _update_equity_curve(self, trade: Dict):
        """Update equity curve with new trade."""
        try:
            # Get trade date and P&L
            date = pd.to_datetime(trade['exit_time'])
            pnl = trade['pnl']
            
            # Add to equity curve
            if date in self.equity_curve.index:
                self.equity_curve[date] += pnl
            else:
                self.equity_curve[date] = pnl
                
            # Calculate cumulative equity
            self.equity_curve = self.equity_curve.cumsum()
            
            # Sort index
            self.equity_curve = self.equity_curve.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {str(e)}")
            
    def analyze_performance(self) -> Dict:
        """Analyze trading performance and generate metrics."""
        try:
            # Calculate basic metrics
            metrics = self._calculate_basic_metrics()
            
            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics()
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics()
            
            # Combine results
            analysis = {
                'basic_metrics': metrics,
                'advanced_metrics': advanced_metrics,
                'trade_statistics': trade_stats
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {
                'basic_metrics': {},
                'advanced_metrics': {},
                'trade_statistics': {}
            }
            
    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic performance metrics."""
        try:
            if self.daily_returns.empty:
                return {}
                
            # Calculate returns
            total_return = self.daily_returns.sum()
            avg_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            
            # Calculate win rate
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            return {
                'total_return': total_return,
                'average_return': avg_return,
                'return_std': std_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}
            
    def _calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced performance metrics."""
        try:
            if self.daily_returns.empty:
                return {}
                
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = self._calculate_sharpe_ratio(risk_free_rate)
            
            # Calculate Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(risk_free_rate)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate Calmar ratio
            calmar_ratio = self._calculate_calmar_ratio()
            
            # Calculate information ratio
            information_ratio = self._calculate_information_ratio()
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {}
            
    def _calculate_trade_statistics(self) -> Dict:
        """Calculate detailed trade statistics."""
        try:
            if not self.trades:
                return {}
                
            # Calculate trade metrics
            pnls = [t['pnl'] for t in self.trades]
            durations = [(pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).total_seconds() / 3600 for t in self.trades]
            
            # Calculate statistics
            stats = {
                'average_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0.0,
                'average_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0.0,
                'largest_win': max(pnls) if pnls else 0.0,
                'largest_loss': min(pnls) if pnls else 0.0,
                'average_duration': np.mean(durations) if durations else 0.0,
                'profit_factor': abs(sum([p for p in pnls if p > 0]) / sum([p for p in pnls if p < 0])) if sum([p for p in pnls if p < 0]) != 0 else float('inf')
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {str(e)}")
            return {}
            
    def _calculate_sharpe_ratio(self, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        try:
            if self.daily_returns.empty:
                return 0.0
                
            # Calculate annualized metrics
            annual_return = self.daily_returns.mean() * 252
            annual_std = self.daily_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            sharpe = (annual_return - risk_free_rate) / annual_std if annual_std != 0 else 0.0
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
            
    def _calculate_sortino_ratio(self, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        try:
            if self.daily_returns.empty:
                return 0.0
                
            # Calculate annualized metrics
            annual_return = self.daily_returns.mean() * 252
            downside_returns = self.daily_returns[self.daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            
            # Calculate Sortino ratio
            sortino = (annual_return - risk_free_rate) / downside_std if downside_std != 0 else 0.0
            
            return sortino
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
            
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if self.equity_curve.empty:
                return 0.0
                
            # Calculate running maximum
            running_max = self.equity_curve.expanding().max()
            
            # Calculate drawdown
            drawdown = (self.equity_curve - running_max) / running_max
            
            # Get maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return 0.0
            
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        try:
            if self.daily_returns.empty:
                return 0.0
                
            # Calculate annualized return
            annual_return = self.daily_returns.mean() * 252
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate Calmar ratio
            calmar = annual_return / max_drawdown if max_drawdown != 0 else 0.0
            
            return calmar
            
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0
            
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio."""
        try:
            if self.daily_returns.empty:
                return 0.0
                
            # Calculate tracking error
            benchmark_returns = pd.Series(0.0, index=self.daily_returns.index)  # Assuming 0% benchmark
            tracking_error = (self.daily_returns - benchmark_returns).std() * np.sqrt(252)
            
            # Calculate information ratio
            information_ratio = (self.daily_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0.0
            
            return information_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating information ratio: {str(e)}")
            return 0.0
            
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot performance metrics and charts."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot equity curve
            ax1 = plt.subplot(2, 2, 1)
            self._plot_equity_curve(ax1)
            
            # Plot drawdown
            ax2 = plt.subplot(2, 2, 2)
            self._plot_drawdown(ax2)
            
            # Plot monthly returns
            ax3 = plt.subplot(2, 2, 3)
            self._plot_monthly_returns(ax3)
            
            # Plot trade distribution
            ax4 = plt.subplot(2, 2, 4)
            self._plot_trade_distribution(ax4)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path)
                
            # Show plot
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting performance: {str(e)}")
            
    def _plot_equity_curve(self, ax):
        """Plot equity curve."""
        try:
            if not self.equity_curve.empty:
                self.equity_curve.plot(ax=ax)
                ax.set_title('Equity Curve')
                ax.set_xlabel('Date')
                ax.set_ylabel('Equity')
                ax.grid(True)
                
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {str(e)}")
            
    def _plot_drawdown(self, ax):
        """Plot drawdown chart."""
        try:
            if not self.equity_curve.empty:
                # Calculate drawdown
                running_max = self.equity_curve.expanding().max()
                drawdown = (self.equity_curve - running_max) / running_max
                
                # Plot drawdown
                drawdown.plot(ax=ax)
                ax.set_title('Drawdown')
                ax.set_xlabel('Date')
                ax.set_ylabel('Drawdown')
                ax.grid(True)
                
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {str(e)}")
            
    def _plot_monthly_returns(self, ax):
        """Plot monthly returns heatmap."""
        try:
            if not self.daily_returns.empty:
                # Calculate monthly returns
                monthly_returns = self.daily_returns.resample('M').sum()
                
                # Create monthly returns matrix
                returns_matrix = monthly_returns.values.reshape(-1, 12)
                
                # Plot heatmap
                sns.heatmap(returns_matrix, ax=ax, cmap='RdYlGn', center=0)
                ax.set_title('Monthly Returns')
                ax.set_xlabel('Month')
                ax.set_ylabel('Year')
                
        except Exception as e:
            self.logger.error(f"Error plotting monthly returns: {str(e)}")
            
    def _plot_trade_distribution(self, ax):
        """Plot trade P&L distribution."""
        try:
            if self.trades:
                # Get trade P&Ls
                pnls = [t['pnl'] for t in self.trades]
                
                # Plot histogram
                sns.histplot(pnls, ax=ax, kde=True)
                ax.set_title('Trade P&L Distribution')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Frequency')
                
        except Exception as e:
            self.logger.error(f"Error plotting trade distribution: {str(e)}")
            
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate performance report."""
        try:
            # Get performance analysis
            analysis = self.analyze_performance()
            
            # Generate report text
            report = []
            report.append("Trading Performance Report")
            report.append("=" * 50)
            report.append("")
            
            # Basic metrics
            report.append("Basic Metrics")
            report.append("-" * 20)
            for key, value in analysis['basic_metrics'].items():
                report.append(f"{key}: {value:.4f}")
            report.append("")
            
            # Advanced metrics
            report.append("Advanced Metrics")
            report.append("-" * 20)
            for key, value in analysis['advanced_metrics'].items():
                report.append(f"{key}: {value:.4f}")
            report.append("")
            
            # Trade statistics
            report.append("Trade Statistics")
            report.append("-" * 20)
            for key, value in analysis['trade_statistics'].items():
                report.append(f"{key}: {value:.4f}")
            report.append("")
            
            # Join report lines
            report_text = "\n".join(report)
            
            # Save report if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                    
            return report_text
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return "Error generating performance report" 