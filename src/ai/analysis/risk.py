import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from scipy import stats
import json
import os

class RiskAnalyzer:
    """Analyzes and manages trading risk."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk parameters
        self.max_position_size = config.get('max_position_size', 0.02)  # 2% of capital
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10% max drawdown
        self.min_risk_reward = config.get('min_risk_reward', 2.0)  # Minimum 2:1 reward/risk
        self.max_open_trades = config.get('max_open_trades', 3)  # Maximum concurrent trades
        
        # Initialize risk metrics
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_daily_pnl = 0.0
        self.max_drawdown_so_far = 0.0
        self.open_trades = 0
        
        # Create cache directory
        self.cache_dir = "cache/risk"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load historical risk metrics
        self._load_risk_metrics()
        
    def _load_risk_metrics(self):
        """Load historical risk metrics from cache."""
        try:
            metrics_file = os.path.join(self.cache_dir, "risk_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    self.daily_pnl = metrics.get('daily_pnl', 0.0)
                    self.total_pnl = metrics.get('total_pnl', 0.0)
                    self.max_daily_pnl = metrics.get('max_daily_pnl', 0.0)
                    self.max_drawdown_so_far = metrics.get('max_drawdown_so_far', 0.0)
                    self.open_trades = metrics.get('open_trades', 0)
                    
        except Exception as e:
            self.logger.error(f"Error loading risk metrics: {str(e)}")
            
    def _save_risk_metrics(self):
        """Save current risk metrics to cache."""
        try:
            metrics = {
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'max_daily_pnl': self.max_daily_pnl,
                'max_drawdown_so_far': self.max_drawdown_so_far,
                'open_trades': self.open_trades,
                'timestamp': datetime.now().isoformat()
            }
            
            metrics_file = os.path.join(self.cache_dir, "risk_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
                
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {str(e)}")
            
    def analyze_risk(self, data: pd.DataFrame, position: Dict) -> Dict:
        """Analyze risk for a potential trade."""
        try:
            # Calculate basic risk metrics
            volatility = self._calculate_volatility(data)
            var = self._calculate_var(data)
            expected_shortfall = self._calculate_expected_shortfall(data)
            
            # Calculate position-specific risk
            position_risk = self._calculate_position_risk(data, position)
            
            # Check risk limits
            risk_limits = self._check_risk_limits()
            
            # Calculate optimal position size
            optimal_size = self._calculate_optimal_position_size(
                data,
                position,
                volatility,
                var
            )
            
            # Combine results
            risk_analysis = {
                'volatility': volatility,
                'var': var,
                'expected_shortfall': expected_shortfall,
                'position_risk': position_risk,
                'risk_limits': risk_limits,
                'optimal_size': optimal_size
            }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk: {str(e)}")
            return {
                'volatility': 0.0,
                'var': 0.0,
                'expected_shortfall': 0.0,
                'position_risk': {},
                'risk_limits': {},
                'optimal_size': 0.0
            }
            
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate price volatility."""
        try:
            returns = data['close'].pct_change()
            return returns.rolling(window=window).std().iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
            
    def _calculate_var(self, data: pd.DataFrame, confidence: float = 0.95, window: int = 20) -> float:
        """Calculate Value at Risk."""
        try:
            returns = data['close'].pct_change()
            return np.percentile(returns.dropna(), (1 - confidence) * 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
            
    def _calculate_expected_shortfall(self, data: pd.DataFrame, confidence: float = 0.95, window: int = 20) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        try:
            returns = data['close'].pct_change()
            var = self._calculate_var(data, confidence, window)
            return returns[returns <= var].mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0
            
    def _calculate_position_risk(self, data: pd.DataFrame, position: Dict) -> Dict:
        """Calculate risk metrics for a specific position."""
        try:
            # Get position details
            entry_price = position.get('entry_price', 0.0)
            stop_loss = position.get('stop_loss', 0.0)
            take_profit = position.get('take_profit', 0.0)
            size = position.get('size', 0.0)
            
            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss) * size
            reward_amount = abs(take_profit - entry_price) * size
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
            
            # Calculate probability of success
            success_prob = self._calculate_success_probability(data, position)
            
            return {
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_reward_ratio': risk_reward_ratio,
                'success_probability': success_prob
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {str(e)}")
            return {
                'risk_amount': 0.0,
                'reward_amount': 0.0,
                'risk_reward_ratio': 0.0,
                'success_probability': 0.0
            }
            
    def _calculate_success_probability(self, data: pd.DataFrame, position: Dict) -> float:
        """Calculate probability of trade success."""
        try:
            # Get position details
            entry_price = position.get('entry_price', 0.0)
            stop_loss = position.get('stop_loss', 0.0)
            take_profit = position.get('take_profit', 0.0)
            
            # Calculate price movements
            returns = data['close'].pct_change()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Calculate probabilities
            loss_threshold = (stop_loss - entry_price) / entry_price
            profit_threshold = (take_profit - entry_price) / entry_price
            
            # Use normal distribution to estimate probabilities
            loss_prob = stats.norm.cdf(loss_threshold, mean_return, std_return)
            profit_prob = 1 - stats.norm.cdf(profit_threshold, mean_return, std_return)
            
            return profit_prob / (profit_prob + loss_prob)
            
        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return 0.0
            
    def _check_risk_limits(self) -> Dict:
        """Check if current risk metrics are within limits."""
        try:
            # Check daily loss limit
            daily_loss_limit = self.daily_pnl < -self.max_daily_loss
            
            # Check drawdown limit
            drawdown_limit = self.max_drawdown_so_far < self.max_drawdown
            
            # Check open trades limit
            trades_limit = self.open_trades < self.max_open_trades
            
            return {
                'daily_loss_limit': daily_loss_limit,
                'drawdown_limit': drawdown_limit,
                'trades_limit': trades_limit,
                'can_trade': all([daily_loss_limit, drawdown_limit, trades_limit])
            }
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return {
                'daily_loss_limit': False,
                'drawdown_limit': False,
                'trades_limit': False,
                'can_trade': False
            }
            
    def _calculate_optimal_position_size(self, data: pd.DataFrame, position: Dict, volatility: float, var: float) -> float:
        """Calculate optimal position size based on risk parameters."""
        try:
            # Get account equity
            equity = position.get('equity', 0.0)
            
            # Calculate base position size
            base_size = equity * self.max_position_size
            
            # Adjust for volatility
            vol_factor = 1.0 / (1.0 + volatility)
            
            # Adjust for VaR
            var_factor = 1.0 / (1.0 + abs(var))
            
            # Adjust for current risk metrics
            risk_factor = 1.0
            if self.daily_pnl < 0:
                risk_factor *= 0.5
            if self.max_drawdown_so_far > self.max_drawdown * 0.5:
                risk_factor *= 0.5
                
            # Calculate final position size
            optimal_size = base_size * vol_factor * var_factor * risk_factor
            
            # Ensure position size is within limits
            optimal_size = min(optimal_size, equity * self.max_position_size)
            optimal_size = max(optimal_size, 0.0)
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {str(e)}")
            return 0.0
            
    def update_metrics(self, pnl: float):
        """Update risk metrics with new P&L."""
        try:
            # Update P&L metrics
            self.daily_pnl += pnl
            self.total_pnl += pnl
            
            # Update maximum daily P&L
            self.max_daily_pnl = max(self.max_daily_pnl, self.daily_pnl)
            
            # Update drawdown
            if self.total_pnl > 0:
                current_drawdown = (self.max_daily_pnl - self.daily_pnl) / self.max_daily_pnl
                self.max_drawdown_so_far = max(self.max_drawdown_so_far, current_drawdown)
                
            # Save updated metrics
            self._save_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
            
    def reset_daily_metrics(self):
        """Reset daily risk metrics."""
        try:
            self.daily_pnl = 0.0
            self.max_daily_pnl = 0.0
            self._save_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Error resetting daily metrics: {str(e)}")
            
    def update_open_trades(self, count: int):
        """Update number of open trades."""
        try:
            self.open_trades = count
            self._save_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating open trades: {str(e)}")
            
    def get_risk_summary(self) -> Dict:
        """Get summary of current risk metrics."""
        try:
            return {
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'max_daily_pnl': self.max_daily_pnl,
                'max_drawdown_so_far': self.max_drawdown_so_far,
                'open_trades': self.open_trades,
                'risk_limits': self._check_risk_limits()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {str(e)}")
            return {
                'daily_pnl': 0.0,
                'total_pnl': 0.0,
                'max_daily_pnl': 0.0,
                'max_drawdown_so_far': 0.0,
                'open_trades': 0,
                'risk_limits': {}
            } 