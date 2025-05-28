import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
from utils.risk_utils import get_risk_param
from dataclasses import dataclass

@dataclass
class RiskLimits:
    max_position_size: float
    max_daily_trades: int
    max_daily_loss: float
    max_drawdown: float
    min_risk_reward: float
    max_open_trades: int
    base_risk_per_trade: float

class RiskManager:
    """Advanced risk management system combining quantitative and practical features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.risk_config = config['risk']
        
        # Initialize risk limits
        self.risk_limits = RiskLimits(
            max_position_size=self.risk_config.get('max_position_size', 0.03),
            max_daily_trades=self.risk_config.get('max_daily_trades', 10),
            max_daily_loss=self.risk_config.get('max_daily_loss', 0.02),
            max_drawdown=self.risk_config.get('max_drawdown', 0.15),
            min_risk_reward=self.risk_config.get('min_risk_reward', 2.0),
            max_open_trades=self.risk_config.get('max_open_trades', 3),
            base_risk_per_trade=self.risk_config.get('base_risk_per_trade', 0.01)
        )
        
        # Portfolio tracking
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.positions = {}
        self.trade_history = []
        self.daily_trades = {}
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.portfolio_exposure = 0.0
        self.position_correlations = {}
        
        self.logger.info("Risk manager initialized")
        
    def initialize(self, initial_balance: float):
        """Initialize the risk manager with starting balance."""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.logger.info(f"Risk manager initialized with balance: {initial_balance}")
        
    def calculate_position_size(self, symbol: str, signals: Dict) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Get current price and stop loss
            current_price = signals.get('price', 0)
            stop_loss = self.calculate_stop_loss(symbol, signals)
            
            if current_price <= 0 or stop_loss <= 0:
                return 0
            
            # Calculate risk amount
            risk_amount = self.current_balance * self.risk_limits.base_risk_per_trade
            
            # Calculate position size
            risk_per_unit = abs(current_price - stop_loss)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Apply maximum position size limit
            max_size = self.current_balance * self.risk_limits.max_position_size
            position_size = min(position_size, max_size)
            
            # Round to appropriate precision
            position_size = round(position_size, 8)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_stop_loss(self, symbol: str, signals: Dict) -> float:
        """Calculate stop loss level."""
        try:
            current_price = signals.get('price', 0)
            if current_price <= 0:
                return 0
            
            # Get stop loss percentage from config
            stop_loss_pct = self.risk_config.get('stop_loss', 0.015)
            
            # Calculate stop loss
            if signals.get('action') == 'buy':
                return current_price * (1 - stop_loss_pct)
            else:
                return current_price * (1 + stop_loss_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return 0

    def calculate_take_profit(self, symbol: str, signals: Dict) -> float:
        """Calculate take profit level."""
        try:
            current_price = signals.get('price', 0)
            if current_price <= 0:
                return 0
            
            # Get take profit percentage from config
            take_profit_pct = self.risk_config.get('take_profit', 0.03)
            
            # Calculate take profit
            if signals.get('action') == 'buy':
                return current_price * (1 + take_profit_pct)
            else:
                return current_price * (1 - take_profit_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return 0

    def validate_trade(self, 
                      symbol: str,
                      side: str,
                      size: float,
                      price: float,
                      stop_loss: float,
                      take_profit: float) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria."""
        try:
            # Check daily trade limit
            if not self._check_daily_trade_limit(symbol):
                return False, "Daily trade limit reached"
            
            # Check portfolio exposure
            if not self._check_portfolio_exposure(symbol, size, price):
                return False, "Portfolio exposure limit exceeded"
            
            # Check position correlation
            if not self._validate_correlation(symbol):
                return False, "High correlation with existing positions"
            
            # Check risk:reward ratio
            if not self._check_risk_reward_ratio(stop_loss, take_profit):
                return False, "Insufficient risk:reward ratio"
            
            # Check margin requirements
            if not self._check_margin_requirements(symbol, size, price):
                return False, "Insufficient margin"
            
            # Check volatility
            if not self._check_volatility(symbol):
                return False, "Volatility too high"
            
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    def update_position_risk(self, 
                           symbol: str,
                           position: Dict[str, Any],
                           current_price: float):
        """Update risk metrics for an open position."""
        try:
            # Calculate current PnL
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            pnl = (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
            
            # Update trailing stop if needed
            if get_risk_param(self.config, symbol, 'trailing_stop')['enabled']:
                self._update_trailing_stop(position, pnl)
            
            # Update position correlation
            self._update_position_correlation(symbol)
            
            # Update portfolio exposure
            self._update_portfolio_exposure()
            
            # Update drawdown
            self._update_drawdown()
            
            # Check if position should be closed
            should_close, reason = self._check_position_exit(position, pnl)
            if should_close:
                self.logger.info(f"Position {symbol} should be closed: {reason}")
                return True, reason
                
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Error updating position risk: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def _estimate_win_rate(self, symbol: str) -> float:
        """Estimate win rate based on historical performance."""
        try:
            if not self.trade_history:
                return 0.5  # Conservative default
                
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol]
            if not symbol_trades:
                return 0.5
                
            wins = sum(1 for t in symbol_trades if t['pnl'] > 0)
            return wins / len(symbol_trades)
            
        except Exception as e:
            self.logger.error(f"Error estimating win rate: {str(e)}")
            return 0.5
            
    def _calculate_avg_win(self, symbol: str) -> float:
        """Calculate average winning trade size."""
        try:
            if not self.trade_history:
                return 0.02  # Conservative default
                
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['pnl'] > 0]
            if not symbol_trades:
                return 0.02
                
            return sum(t['pnl'] for t in symbol_trades) / len(symbol_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating average win: {str(e)}")
            return 0.02
            
    def _calculate_avg_loss(self, symbol: str) -> float:
        """Calculate average losing trade size."""
        try:
            if not self.trade_history:
                return 0.01  # Conservative default
                
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['pnl'] < 0]
            if not symbol_trades:
                return 0.01
                
            return abs(sum(t['pnl'] for t in symbol_trades) / len(symbol_trades))
            
        except Exception as e:
            self.logger.error(f"Error calculating average loss: {str(e)}")
            return 0.01
            
    def _check_correlation(self, symbol: str) -> float:
        """Check correlation with existing positions."""
        try:
            if not self.positions:
                return 0.0
                
            # Get price data for correlation calculation
            symbol_data = self._get_price_data(symbol)
            if symbol_data is None:
                return 0.0
                
            max_correlation = 0.0
            for pos_symbol in self.positions:
                pos_data = self._get_price_data(pos_symbol)
                if pos_data is not None:
                    correlation = symbol_data.corr(pos_data)
                    max_correlation = max(max_correlation, abs(correlation))
                    
            return max_correlation
            
        except Exception as e:
            self.logger.error(f"Error checking correlation: {str(e)}")
            return 0.0
            
    def _detect_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Detect current market regime."""
        try:
            # Get price data
            price_data = self._get_price_data(symbol)
            if price_data is None:
                return {'regime': 'unknown', 'confidence': 0.0}
                
            # Calculate volatility
            returns = price_data.pct_change()
            volatility = returns.std()
            
            # Calculate trend strength
            sma_20 = price_data.rolling(20).mean()
            sma_50 = price_data.rolling(50).mean()
            trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # Determine regime
            if trend_strength > 0.1:
                regime = 'trending'
                confidence = min(trend_strength * 5, 1.0)
            elif volatility > 0.02:
                regime = 'volatile'
                confidence = min(volatility * 10, 1.0)
            else:
                regime = 'ranging'
                confidence = 0.7
                
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.0}
            
    def _get_regime_multiplier(self, regime: Dict[str, Any]) -> float:
        """Get position size multiplier based on market regime."""
        try:
            regime_config = self.config.get('market_regime', {})
            regime_params = regime_config.get(regime['regime'], {})
            return regime_params.get('position_multiplier', 1.0)
            
        except Exception as e:
            self.logger.error(f"Error getting regime multiplier: {str(e)}")
            return 1.0
            
    def _check_daily_trade_limit(self, symbol: str) -> bool:
        """Check if daily trade limit is reached."""
        try:
            today = datetime.now().date()
            daily_trades = self.daily_trades.get(today, {}).get(symbol, 0)
            max_daily_trades = get_risk_param(self.config, symbol, 'max_daily_trades')
            return daily_trades < max_daily_trades
            
        except Exception as e:
            self.logger.error(f"Error checking daily trade limit: {str(e)}")
            return False
            
    def _check_portfolio_exposure(self, symbol: str, size: float, price: float) -> bool:
        """Check if portfolio exposure limit is exceeded."""
        try:
            new_exposure = self.portfolio_exposure + (size * price)
            max_exposure = self.current_balance * get_risk_param(self.config, symbol, 'max_portfolio_risk')
            return new_exposure <= max_exposure
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio exposure: {str(e)}")
            return False
            
    def _validate_correlation(self, symbol: str) -> bool:
        """Validate if correlation with existing positions is acceptable."""
        try:
            correlation = self._check_correlation(symbol)
            max_correlation = get_risk_param(self.config, symbol, 'max_correlation')
            return correlation <= max_correlation
            
        except Exception as e:
            self.logger.error(f"Error validating correlation: {str(e)}")
            return False
            
    def _check_risk_reward_ratio(self, stop_loss: float, take_profit: float) -> bool:
        """Check if risk:reward ratio meets minimum requirement."""
        try:
            min_ratio = get_risk_param(self.config, symbol, 'min_risk_reward_ratio')
            return take_profit / stop_loss >= min_ratio
            
        except Exception as e:
            self.logger.error(f"Error checking risk:reward ratio: {str(e)}")
            return False
            
    def _check_margin_requirements(self, symbol: str, size: float, price: float) -> bool:
        """Check if margin requirements are met."""
        try:
            required_margin = size * price * get_risk_param(self.config, symbol, 'margin_requirement')
            available_margin = self.current_balance - self.portfolio_exposure
            return available_margin >= required_margin
            
        except Exception as e:
            self.logger.error(f"Error checking margin requirements: {str(e)}")
            return False
            
    def _check_volatility(self, symbol: str) -> bool:
        """Check if volatility is within acceptable range."""
        try:
            regime = self._detect_market_regime(symbol)
            max_volatility = get_risk_param(self.config, symbol, 'max_volatility')
            return regime['volatility'] <= max_volatility
            
        except Exception as e:
            self.logger.error(f"Error checking volatility: {str(e)}")
            return False
            
    def _update_trailing_stop(self, position: Dict[str, Any], pnl: float):
        """Update trailing stop for a position."""
        try:
            trailing_stop_config = get_risk_param(self.config, symbol, 'trailing_stop')
            if pnl > trailing_stop_config['activation']:
                new_stop = pnl - trailing_stop_config['step']
                position['trailing_stop'] = max(position.get('trailing_stop', 0), new_stop)
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
            
    def _update_position_correlation(self, symbol: str):
        """Update correlation matrix for all positions."""
        try:
            if len(self.positions) < 2:
                return
                
            symbols = list(self.positions.keys())
            price_data = {s: self._get_price_data(s) for s in symbols}
            
            for s1 in symbols:
                for s2 in symbols:
                    if s1 != s2 and price_data[s1] is not None and price_data[s2] is not None:
                        correlation = price_data[s1].corr(price_data[s2])
                        self.position_correlations[(s1, s2)] = correlation
                        
        except Exception as e:
            self.logger.error(f"Error updating position correlation: {str(e)}")
            
    def _update_portfolio_exposure(self):
        """Update total portfolio exposure."""
        try:
            self.portfolio_exposure = sum(
                pos['size'] * pos['current_price']
                for pos in self.positions.values()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio exposure: {str(e)}")
            
    def _update_drawdown(self):
        """Update current drawdown."""
        try:
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
                
            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        except Exception as e:
            self.logger.error(f"Error updating drawdown: {str(e)}")
            
    def _check_position_exit(self, position: Dict[str, Any], pnl: float) -> Tuple[bool, str]:
        """Check if position should be exited."""
        try:
            # Check trailing stop
            if 'trailing_stop' in position and pnl < position['trailing_stop']:
                return True, "Trailing stop hit"
                
            # Check stop loss
            if pnl < -get_risk_param(self.config, symbol, 'stop_loss'):
                return True, "Stop loss hit"
                
            # Check take profit
            if pnl > get_risk_param(self.config, symbol, 'take_profit'):
                return True, "Take profit hit"
                
            # Check time-based exit
            if 'entry_time' in position:
                time_in_trade = (datetime.now() - position['entry_time']).total_seconds() / 3600
                if time_in_trade > get_risk_param(self.config, symbol, 'max_trade_duration'):
                    return True, "Maximum trade duration reached"
                    
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Error checking position exit: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def _get_price_data(self, symbol: str) -> Optional[pd.Series]:
        """Get price data for a symbol."""
        try:
            # This should be implemented to fetch actual price data
            # For now, return None
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price data: {str(e)}")
            return None 