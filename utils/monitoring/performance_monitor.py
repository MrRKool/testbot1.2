import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3
import pandas as pd
import numpy as np

class PerformanceMonitor:
    """Class voor het monitoren van trading performance metrics."""
    
    def __init__(self, report_dir: str = 'reports', metrics_window: int = 1000, report_interval: int = 86400):
        """Initialize de performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.report_dir = report_dir
        self.metrics_window = metrics_window
        self.report_interval = report_interval
        self.last_report_time = time.time()
        
        # Initialize metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0
        }
        
        # Create reports directory if it doesn't exist
        import os
        os.makedirs(report_dir, exist_ok=True)
        
        self.logger.info("Performance monitor initialized")
        
    def update_metrics(self, trade_data: Optional[Dict] = None):
        """Update performance metrics with new trade data."""
        try:
            if trade_data:
                # Update basic metrics
                self.metrics['total_trades'] += 1
                if trade_data.get('pnl', 0) > 0:
                    self.metrics['winning_trades'] += 1
                    self.metrics['total_profit'] += trade_data['pnl']
                else:
                    self.metrics['losing_trades'] += 1
                    self.metrics['total_loss'] += abs(trade_data['pnl'])
                
                # Calculate derived metrics
                self._calculate_derived_metrics()
                
                # Save metrics to database
                self._save_metrics()
                
                # Generate report if needed
                if time.time() - self.last_report_time >= self.report_interval:
                    self._generate_report()
                    self.last_report_time = time.time()
                    
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    def _calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        try:
            # Win rate
            if self.metrics['total_trades'] > 0:
                self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
            
            # Profit factor
            if self.metrics['total_loss'] > 0:
                self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
            
            # Average win/loss
            if self.metrics['winning_trades'] > 0:
                self.metrics['average_win'] = self.metrics['total_profit'] / self.metrics['winning_trades']
            if self.metrics['losing_trades'] > 0:
                self.metrics['average_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']
            
            # Calculate Sharpe and Sortino ratios
            self._calculate_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {str(e)}")
            
    def _calculate_risk_metrics(self):
        """Calculate risk-adjusted return metrics."""
        try:
            # Get recent trade returns
            returns = self._get_recent_returns()
            if len(returns) > 0:
                # Sharpe ratio
                excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
                if len(excess_returns) > 1:
                    self.metrics['sharpe_ratio'] = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
                
                # Sortino ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 1:
                    self.metrics['sortino_ratio'] = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = cumulative_returns / rolling_max - 1
                self.metrics['max_drawdown'] = abs(drawdowns.min())
                
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            
    def _get_recent_returns(self) -> np.ndarray:
        """Get recent trade returns from database."""
        try:
            with sqlite3.connect('trading_bot.db') as conn:
                query = """
                    SELECT pnl FROM trades 
                    WHERE status = 'closed' 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                returns = pd.read_sql_query(query, conn, params=(self.metrics_window,))
                return returns['pnl'].values
        except Exception as e:
            self.logger.error(f"Error getting recent returns: {str(e)}")
            return np.array([])
            
    def _save_metrics(self):
        """Save current metrics to database."""
        try:
            with sqlite3.connect('trading_bot.db') as conn:
                conn.execute("""
                    INSERT INTO performance_metrics (
                        total_trades, winning_trades, losing_trades,
                        win_rate, profit_factor, sharpe_ratio,
                        sortino_ratio, max_drawdown, average_win,
                        average_loss, total_profit, total_loss,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.metrics['total_trades'],
                    self.metrics['winning_trades'],
                    self.metrics['losing_trades'],
                    self.metrics['win_rate'],
                    self.metrics['profit_factor'],
                    self.metrics['sharpe_ratio'],
                    self.metrics['sortino_ratio'],
                    self.metrics['max_drawdown'],
                    self.metrics['average_win'],
                    self.metrics['average_loss'],
                    self.metrics['total_profit'],
                    self.metrics['total_loss'],
                    json.dumps(self.metrics)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def _generate_report(self):
        """Generate performance report."""
        try:
            report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"{self.report_dir}/performance_report_{report_time}.json"
            
            with open(report_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
                
            self.logger.info(f"Performance report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy() 