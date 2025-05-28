import sys
import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from bot.risk.risk_manager import RiskManager
from bot.analysis.market_regime import MarketRegimeDetector

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    var_95: float
    expected_shortfall: float
    regime_performance: Dict[str, float]
    correlation_matrix: pd.DataFrame
    beta: float
    alpha: float
    information_ratio: float

class PerformanceAnalytics:
    """Real-time performance analytics system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance analytics with configuration."""
        self.config = config
        self.risk_manager = RiskManager(config)
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.lookback_period = 100
        self.risk_free_rate = 0.02
        self.historical_returns = []
        self.historical_pnl = []
        self.trade_history = []
        
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade to the analytics."""
        self.trades.append(trade)
        
    def add_daily_return(self, return_pct: float):
        """Add daily return percentage."""
        self.daily_returns.append(return_pct)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {}
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
        
        # Risk metrics
        returns = pd.Series(self.daily_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def generate_report(self) -> str:
        """Generate a performance report."""
        metrics = self.calculate_metrics()
        
        report = "Performance Report\n"
        report += "=================\n\n"
        
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Win Rate: {metrics['win_rate']:.2%}\n"
        report += f"Total PnL: {metrics['total_pnl']:.2f} USDT\n"
        report += f"Average Win: {metrics['avg_win']:.2f} USDT\n"
        report += f"Average Loss: {metrics['avg_loss']:.2f} USDT\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
        
        return report
        
    def update_metrics(self,
                      portfolio_data: Dict,
                      market_data: Dict[str, pd.DataFrame],
                      trade_history: List[Dict]) -> PerformanceMetrics:
        """Update and calculate performance metrics."""
        try:
            # Update historical data
            self._update_historical_data(portfolio_data, trade_history)
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                total_return=self._calculate_total_return(),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                sortino_ratio=self._calculate_sortino_ratio(),
                max_drawdown=self._calculate_max_drawdown(),
                win_rate=self._calculate_win_rate(),
                profit_factor=self._calculate_profit_factor(),
                avg_trade=self._calculate_avg_trade(),
                avg_win=self._calculate_avg_win(),
                avg_loss=self._calculate_avg_loss(),
                var_95=self._calculate_var(0.95),
                expected_shortfall=self._calculate_expected_shortfall(),
                regime_performance=self._calculate_regime_performance(market_data),
                correlation_matrix=self._calculate_correlation_matrix(market_data),
                beta=self._calculate_beta(market_data),
                alpha=self._calculate_alpha(market_data),
                information_ratio=self._calculate_information_ratio(market_data)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return None
            
    def _update_historical_data(self,
                              portfolio_data: Dict,
                              trade_history: List[Dict]):
        """Update historical performance data."""
        try:
            # Update returns
            if 'daily_return' in portfolio_data:
                self.historical_returns.append(portfolio_data['daily_return'])
                
            # Update PnL
            if 'daily_pnl' in portfolio_data:
                self.historical_pnl.append(portfolio_data['daily_pnl'])
                
            # Update trade history
            self.trade_history.extend(trade_history)
            
            # Keep only recent data
            if len(self.historical_returns) > self.lookback_period:
                self.historical_returns = self.historical_returns[-self.lookback_period:]
            if len(self.historical_pnl) > self.lookback_period:
                self.historical_pnl = self.historical_pnl[-self.lookback_period:]
                
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            
    def _calculate_total_return(self) -> float:
        """Calculate total return."""
        try:
            if not self.historical_returns:
                return 0.0
            return (1 + np.array(self.historical_returns)).prod() - 1
        except Exception as e:
            logger.error(f"Error calculating total return: {e}")
            return 0.0
            
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(self.historical_returns) < 2:
                return 0.0
                
            returns = np.array(self.historical_returns)
            excess_returns = returns - self.risk_free_rate / 252
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
                
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            
            return np.sqrt(252) * excess_returns.mean() / downside_std
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
            
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        try:
            if not self.trade_history:
                return 0.0
                
            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            return winning_trades / len(self.trade_history)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
            
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        try:
            if not self.trade_history:
                return 0.0
                
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
            
            if not losing_trades:
                return float('inf')
                
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
            
    def _calculate_avg_trade(self) -> float:
        """Calculate average trade PnL."""
        try:
            if not self.trade_history:
                return 0.0
            return np.mean([t['pnl'] for t in self.trade_history])
        except Exception as e:
            logger.error(f"Error calculating average trade: {e}")
            return 0.0
            
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade."""
        try:
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            return np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        except Exception as e:
            logger.error(f"Error calculating average win: {e}")
            return 0.0
            
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade."""
        try:
            losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
            return np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        except Exception as e:
            logger.error(f"Error calculating average loss: {e}")
            return 0.0
            
    def _calculate_var(self, confidence: float) -> float:
        """Calculate Value at Risk."""
        try:
            if len(self.historical_returns) < 2:
                return 0.0
                
            returns = np.array(self.historical_returns)
            return np.percentile(returns, (1 - confidence) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
            
    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        try:
            if len(self.historical_returns) < 2:
                return 0.0
                
            returns = np.array(self.historical_returns)
            var_95 = self._calculate_var(0.95)
            
            return np.mean(returns[returns <= var_95])
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0.0
            
    def _calculate_regime_performance(self,
                                    market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate performance in different market regimes."""
        try:
            regime_returns = {
                'trending': [],
                'ranging': [],
                'volatile': [],
                'reversal': []
            }
            
            for trade in self.trade_history:
                if 'entry_time' not in trade or 'exit_time' not in trade:
                    continue
                    
                # Get market regime during trade
                regime = self.market_regime_detector.detect_regime(
                    self._get_market_data(
                        trade['entry_time'],
                        trade['exit_time'],
                        market_data
                    )
                )
                
                # Add trade return to appropriate regime
                for regime_name, prob in regime.items():
                    if prob > 0.5:  # Consider dominant regime
                        regime_returns[regime_name].append(trade['pnl'])
                        
            # Calculate average returns per regime
            return {
                regime: np.mean(returns) if returns else 0
                for regime, returns in regime_returns.items()
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime performance: {e}")
            return {}
            
    def _calculate_correlation_matrix(self,
                                    market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between assets."""
        try:
            if not market_data:
                return pd.DataFrame()
                
            # Calculate returns for each asset
            returns = pd.DataFrame({
                symbol: data['close'].pct_change()
                for symbol, data in market_data.items()
            })
            
            # Calculate correlation matrix
            return returns.corr()
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
            
    def _calculate_beta(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio beta."""
        try:
            if not market_data or not self.historical_returns:
                return 0.0
                
            # Get market returns (assuming first asset is market)
            market_returns = next(iter(market_data.values()))['close'].pct_change()
            
            # Calculate covariance and variance
            covariance = np.cov(self.historical_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0
            
    def _calculate_alpha(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio alpha."""
        try:
            if not market_data or not self.historical_returns:
                return 0.0
                
            # Get market returns
            market_returns = next(iter(market_data.values()))['close'].pct_change()
            
            # Calculate beta
            beta = self._calculate_beta(market_data)
            
            # Calculate alpha
            portfolio_return = np.mean(self.historical_returns)
            market_return = np.mean(market_returns)
            
            return portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0.0
            
    def _calculate_information_ratio(self,
                                   market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate information ratio."""
        try:
            if not market_data or not self.historical_returns:
                return 0.0
                
            # Get market returns
            market_returns = next(iter(market_data.values()))['close'].pct_change()
            
            # Calculate tracking error
            tracking_error = np.std(np.array(self.historical_returns) - market_returns)
            
            # Calculate excess return
            excess_return = np.mean(self.historical_returns) - np.mean(market_returns)
            
            return excess_return / tracking_error if tracking_error > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
            
    def _get_market_data(self,
                        start_time: datetime,
                        end_time: datetime,
                        market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Get market data for a specific time period."""
        try:
            filtered_data = {}
            for symbol, data in market_data.items():
                mask = (data.index >= start_time) & (data.index <= end_time)
                filtered_data[symbol] = data[mask]
            return filtered_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {} 