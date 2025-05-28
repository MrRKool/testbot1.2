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
    
    def __init__(self, config: Dict):
        """Initialize risk manager with optimized settings."""
        self.config = config
        self.log = logging.getLogger(__name__)
        
        # Initialize risk parameters
        self.max_position_size = config['risk']['max_position_size']
        self.max_daily_loss = config['risk']['max_daily_loss']
        self.stop_loss = config['risk']['stop_loss']
        self.take_profit = config['risk']['take_profit']
        self.max_open_trades = config['risk']['max_open_trades']
        
        # Initialize performance tracking
        self._performance_metrics = {
            'daily_pnl': [],
            'drawdown': [],
            'win_rate': [],
            'risk_reward_ratio': []
        }
        
        # Initialize trade tracking
        self._active_trades = {}
        self._trade_history = []
        
        # Initialize risk limits
        self.risk_limits = RiskLimits(
            max_position_size=self.max_position_size,
            max_daily_trades=config['risk']['max_daily_trades'],
            max_daily_loss=self.max_daily_loss,
            max_drawdown=config['risk']['max_drawdown'],
            min_risk_reward=config['risk']['min_risk_reward'],
            max_open_trades=self.max_open_trades,
            base_risk_per_trade=config['risk']['base_risk_per_trade']
        )
        
        # Portfolio tracking
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.positions = {}
        self.daily_trades = {}
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.portfolio_exposure = 0.0
        self.position_correlations = {}
        
        self.log.info("Risk manager initialized")
        
    def initialize(self, initial_balance: float):
        """Initialize the risk manager with starting balance."""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.log.info(f"Risk manager initialized with balance: {initial_balance}")
        
    def calculate_position_size(self, signal: Dict, account_balance: float) -> float:
        """Calculate optimal position size based on risk parameters."""
        try:
            # Get base position size from signal
            base_size = signal.get('position_size', 0)
            
            # Apply risk limits
            max_size = min(
                self.max_position_size,
                account_balance * self.config['risk']['max_risk_per_trade']
            )
            
            # Adjust for volatility
            volatility = signal.get('volatility', 1.0)
            adjusted_size = base_size / volatility
            
            # Ensure position size is within limits
            position_size = min(adjusted_size, max_size)
            
            return position_size
            
        except Exception as e:
            self.log.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def calculate_stop_loss(self, entry_price: float, signal: Dict) -> float:
        """Calculate optimal stop loss level."""
        try:
            # Get base stop loss from signal
            base_stop = signal.get('stop_loss', 0)
            
            # Calculate ATR-based stop loss
            atr = signal.get('atr', 0)
            atr_stop = entry_price - (atr * self.config['risk']['atr_multiplier'])
            
            # Use the more conservative stop loss
            stop_loss = max(base_stop, atr_stop)
            
            # Ensure stop loss is within maximum allowed loss
            max_loss = entry_price * (1 - self.stop_loss)
            stop_loss = max(stop_loss, max_loss)
            
            return stop_loss
            
        except Exception as e:
            self.log.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * (1 - self.stop_loss)
            
    def calculate_take_profit(self, entry_price: float, signal: Dict) -> float:
        """Calculate optimal take profit level."""
        try:
            # Get base take profit from signal
            base_tp = signal.get('take_profit', 0)
            
            # Calculate ATR-based take profit
            atr = signal.get('atr', 0)
            atr_tp = entry_price + (atr * self.config['risk']['atr_multiplier'])
            
            # Use the more conservative take profit
            take_profit = min(base_tp, atr_tp)
            
            # Ensure take profit is within maximum allowed profit
            max_profit = entry_price * (1 + self.take_profit)
            take_profit = min(take_profit, max_profit)
            
            return take_profit
            
        except Exception as e:
            self.log.error(f"Error calculating take profit: {str(e)}")
            return entry_price * (1 + self.take_profit)
            
    def can_open_trade(self, symbol: str) -> bool:
        """Check if a new trade can be opened."""
        try:
            # Check maximum open trades
            if len(self._active_trades) >= self.max_open_trades:
                return False
                
            # Check if symbol already has an open trade
            if symbol in self._active_trades:
                return False
                
            # Check daily loss limit
            daily_pnl = sum(trade['pnl'] for trade in self._trade_history 
                          if trade['timestamp'].date() == datetime.now().date())
            if daily_pnl <= -self.max_daily_loss:
                return False
                
            return True
            
        except Exception as e:
            self.log.error(f"Error checking if trade can be opened: {str(e)}")
            return False
            
    def update_trade(self, symbol: str, trade_data: Dict):
        """Update trade information."""
        try:
            if symbol in self._active_trades:
                self._active_trades[symbol].update(trade_data)
                
                # Check if trade should be closed
                if self._should_close_trade(symbol):
                    self._close_trade(symbol)
                    
        except Exception as e:
            self.log.error(f"Error updating trade: {str(e)}")
            
    def _should_close_trade(self, symbol: str) -> bool:
        """Check if a trade should be closed."""
        try:
            trade = self._active_trades[symbol]
            
            # Check stop loss
            if trade['current_price'] <= trade['stop_loss']:
                return True
                
            # Check take profit
            if trade['current_price'] >= trade['take_profit']:
                return True
                
            # Check trailing stop
            if 'trailing_stop' in trade:
                if trade['current_price'] <= trade['trailing_stop']:
                    return True
                    
            return False
            
        except Exception as e:
            self.log.error(f"Error checking if trade should be closed: {str(e)}")
            return False
            
    def _close_trade(self, symbol: str):
        """Close a trade and update performance metrics."""
        try:
            trade = self._active_trades.pop(symbol)
            
            # Calculate trade metrics
            pnl = (trade['current_price'] - trade['entry_price']) / trade['entry_price']
            duration = datetime.now() - trade['entry_time']
            
            # Update trade history
            self._trade_history.append({
                'symbol': symbol,
                'entry_price': trade['entry_price'],
                'exit_price': trade['current_price'],
                'pnl': pnl,
                'duration': duration,
                'timestamp': datetime.now()
            })
            
            # Update performance metrics
            self._update_performance_metrics(pnl)
            
        except Exception as e:
            self.log.error(f"Error closing trade: {str(e)}")
            
    def _update_performance_metrics(self, pnl: float):
        """Update performance metrics."""
        try:
            # Update daily PnL
            self._performance_metrics['daily_pnl'].append(pnl)
            
            # Calculate drawdown
            cumulative_pnl = sum(self._performance_metrics['daily_pnl'])
            peak = max(cumulative_pnl, 0)
            drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
            self._performance_metrics['drawdown'].append(drawdown)
            
            # Calculate win rate
            wins = sum(1 for p in self._performance_metrics['daily_pnl'] if p > 0)
            total = len(self._performance_metrics['daily_pnl'])
            win_rate = wins / total if total > 0 else 0
            self._performance_metrics['win_rate'].append(win_rate)
            
            # Calculate risk-reward ratio
            avg_win = np.mean([p for p in self._performance_metrics['daily_pnl'] if p > 0]) if any(p > 0 for p in self._performance_metrics['daily_pnl']) else 0
            avg_loss = np.mean([p for p in self._performance_metrics['daily_pnl'] if p < 0]) if any(p < 0 for p in self._performance_metrics['daily_pnl']) else 0
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            self._performance_metrics['risk_reward_ratio'].append(risk_reward)
            
        except Exception as e:
            self.log.error(f"Error updating performance metrics: {str(e)}")
            
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        try:
            return {
                'daily_pnl': self._performance_metrics['daily_pnl'][-1] if self._performance_metrics['daily_pnl'] else 0,
                'drawdown': self._performance_metrics['drawdown'][-1] if self._performance_metrics['drawdown'] else 0,
                'win_rate': self._performance_metrics['win_rate'][-1] if self._performance_metrics['win_rate'] else 0,
                'risk_reward_ratio': self._performance_metrics['risk_reward_ratio'][-1] if self._performance_metrics['risk_reward_ratio'] else 0
            }
        except Exception as e:
            self.log.error(f"Error getting performance metrics: {str(e)}")
            return {}

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
            self.log.error(f"Error validating trade: {str(e)}")
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
                self.log.info(f"Position {symbol} should be closed: {reason}")
                return True, reason
                
            return False, ""
            
        except Exception as e:
            self.log.error(f"Error updating position risk: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def _estimate_win_rate(self, symbol: str) -> float:
        """Estimate win rate based on historical performance."""
        try:
            if not self._trade_history:
                return 0.5  # Conservative default
                
            symbol_trades = [t for t in self._trade_history if t['symbol'] == symbol]
            if not symbol_trades:
                return 0.5
                
            wins = sum(1 for t in symbol_trades if t['pnl'] > 0)
            return wins / len(symbol_trades)
            
        except Exception as e:
            self.log.error(f"Error estimating win rate: {str(e)}")
            return 0.5
            
    def _calculate_avg_win(self, symbol: str) -> float:
        """Calculate average winning trade size."""
        try:
            if not self._trade_history:
                return 0.02  # Conservative default
                
            symbol_trades = [t for t in self._trade_history if t['symbol'] == symbol and t['pnl'] > 0]
            if not symbol_trades:
                return 0.02
                
            return sum(t['pnl'] for t in symbol_trades) / len(symbol_trades)
            
        except Exception as e:
            self.log.error(f"Error calculating average win: {str(e)}")
            return 0.02
            
    def _calculate_avg_loss(self, symbol: str) -> float:
        """Calculate average losing trade size."""
        try:
            if not self._trade_history:
                return 0.01  # Conservative default
                
            symbol_trades = [t for t in self._trade_history if t['symbol'] == symbol and t['pnl'] < 0]
            if not symbol_trades:
                return 0.01
                
            return abs(sum(t['pnl'] for t in symbol_trades) / len(symbol_trades))
            
        except Exception as e:
            self.log.error(f"Error calculating average loss: {str(e)}")
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
            self.log.error(f"Error checking correlation: {str(e)}")
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
            self.log.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.0}
            
    def _get_regime_multiplier(self, regime: Dict[str, Any]) -> float:
        """Get position size multiplier based on market regime."""
        try:
            regime_config = self.config.get('market_regime', {})
            regime_params = regime_config.get(regime['regime'], {})
            return regime_params.get('position_multiplier', 1.0)
            
        except Exception as e:
            self.log.error(f"Error getting regime multiplier: {str(e)}")
            return 1.0
            
    def _check_daily_trade_limit(self, symbol: str) -> bool:
        """Check if daily trade limit is reached."""
        try:
            today = datetime.now().date()
            daily_trades = self.daily_trades.get(today, {}).get(symbol, 0)
            max_daily_trades = get_risk_param(self.config, symbol, 'max_daily_trades')
            return daily_trades < max_daily_trades
            
        except Exception as e:
            self.log.error(f"Error checking daily trade limit: {str(e)}")
            return False
            
    def _check_portfolio_exposure(self, symbol: str, size: float, price: float) -> bool:
        """Check if portfolio exposure limit is exceeded."""
        try:
            new_exposure = self.portfolio_exposure + (size * price)
            max_exposure = self.current_balance * get_risk_param(self.config, symbol, 'max_portfolio_risk')
            return new_exposure <= max_exposure
            
        except Exception as e:
            self.log.error(f"Error checking portfolio exposure: {str(e)}")
            return False
            
    def _validate_correlation(self, symbol: str) -> bool:
        """Validate if correlation with existing positions is acceptable."""
        try:
            correlation = self._check_correlation(symbol)
            max_correlation = get_risk_param(self.config, symbol, 'max_correlation')
            return correlation <= max_correlation
            
        except Exception as e:
            self.log.error(f"Error validating correlation: {str(e)}")
            return False
            
    def _check_risk_reward_ratio(self, stop_loss: float, take_profit: float) -> bool:
        """Check if risk:reward ratio meets minimum requirement."""
        try:
            min_ratio = get_risk_param(self.config, symbol, 'min_risk_reward_ratio')
            return take_profit / stop_loss >= min_ratio
            
        except Exception as e:
            self.log.error(f"Error checking risk:reward ratio: {str(e)}")
            return False
            
    def _check_margin_requirements(self, symbol: str, size: float, price: float) -> bool:
        """Check if margin requirements are met."""
        try:
            required_margin = size * price * get_risk_param(self.config, symbol, 'margin_requirement')
            available_margin = self.current_balance - self.portfolio_exposure
            return available_margin >= required_margin
            
        except Exception as e:
            self.log.error(f"Error checking margin requirements: {str(e)}")
            return False
            
    def _check_volatility(self, symbol: str) -> bool:
        """Check if volatility is within acceptable range."""
        try:
            regime = self._detect_market_regime(symbol)
            max_volatility = get_risk_param(self.config, symbol, 'max_volatility')
            return regime['volatility'] <= max_volatility
            
        except Exception as e:
            self.log.error(f"Error checking volatility: {str(e)}")
            return False
            
    def _update_trailing_stop(self, position: Dict[str, Any], pnl: float):
        """Update trailing stop for a position."""
        try:
            trailing_stop_config = get_risk_param(self.config, symbol, 'trailing_stop')
            if pnl > trailing_stop_config['activation']:
                new_stop = pnl - trailing_stop_config['step']
                position['trailing_stop'] = max(position.get('trailing_stop', 0), new_stop)
                
        except Exception as e:
            self.log.error(f"Error updating trailing stop: {str(e)}")
            
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
            self.log.error(f"Error updating position correlation: {str(e)}")
            
    def _update_portfolio_exposure(self):
        """Update total portfolio exposure."""
        try:
            self.portfolio_exposure = sum(
                pos['size'] * pos['current_price']
                for pos in self.positions.values()
            )
            
        except Exception as e:
            self.log.error(f"Error updating portfolio exposure: {str(e)}")
            
    def _update_drawdown(self):
        """Update current drawdown."""
        try:
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
                
            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        except Exception as e:
            self.log.error(f"Error updating drawdown: {str(e)}")
            
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
            self.log.error(f"Error checking position exit: {str(e)}")
            return False, f"Error: {str(e)}"
            
    def _get_price_data(self, symbol: str) -> Optional[pd.Series]:
        """Get price data for a symbol."""
        try:
            # This should be implemented to fetch actual price data
            # For now, return None
            return None
            
        except Exception as e:
            self.log.error(f"Error getting price data: {str(e)}")
            return None 