import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import yaml
from functools import lru_cache
import numba
from numba import jit
from utils.risk_utils import get_risk_param
from .indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_ema
from ..ai.ai_learning import AILearningModule

class TradingStrategy:
    def __init__(self, config: dict):
        """Initialize the trading strategy with configuration from config.yaml"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.strategy_config = config['strategy']
        self.timeframes = self.config['timeframes']
        self.data = {}
        self.indicators = {}
        self.positions = {}
        self._indicator_cache = {}
        self._signal_cache = {}
        self._last_signal_time = {}
        self.ai_module = AILearningModule(config)
        
        # Initialize data structures for each symbol
        for symbol in self.config['symbols'].keys():
            self.data[symbol] = {}
            self.indicators[symbol] = {}
            self._last_signal_time[symbol] = {}
            for timeframe in self.timeframes:
                self.data[symbol][timeframe] = pd.DataFrame()
                self.indicators[symbol][timeframe] = pd.DataFrame()
                self._last_signal_time[symbol][timeframe] = datetime.min
        
        self.logger.info("Trading strategy initialized")

    def update_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Update market data for a specific timeframe"""
        if symbol not in self.data:
            self.data[symbol] = {}
            self.indicators[symbol] = {}
            for tf in self.timeframes:
                self.data[symbol][tf] = pd.DataFrame()
                self.indicators[symbol][tf] = pd.DataFrame()
            
        # Update data
        self.data[symbol][timeframe] = data.copy()
        
        # Clear cached indicators for this timeframe
        if timeframe in self.indicators[symbol]:
            self.indicators[symbol][timeframe] = pd.DataFrame()
            if (symbol, timeframe) in self._indicator_cache:
                del self._indicator_cache[(symbol, timeframe)]

    @lru_cache(maxsize=100)
    def calculate_indicators(self, symbol: str, timeframe: str) -> pd.DataFrame:
        cache_key = (symbol, timeframe)
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return pd.DataFrame()
        if self.data[symbol][timeframe].empty:
            return pd.DataFrame()
        df = self.data[symbol][timeframe].copy()
        try:
            indicators_config = self.config['strategy']['indicators']
            df = self._calculate_all_indicators(df, indicators_config)
            self._indicator_cache[cache_key] = df
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        # All indicator parameters are now always from config
        rsi_period = config['rsi']['period']
        rsi_overbought = config['rsi']['overbought']
        rsi_oversold = config['rsi']['oversold']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        macd_fast = config['macd']['fast_period']
        macd_slow = config['macd']['slow_period']
        macd_signal = config['macd']['signal_period']
        df['ema_fast'] = df['close'].ewm(span=macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        bb_period = config['bollinger_bands']['period']
        bb_std = config['bollinger_bands']['std_dev']
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        ema_short = config['ema']['short_period']
        ema_medium = config['ema']['medium_period']
        ema_long = config['ema']['long_period']
        df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=ema_medium, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        # Add additional EMAs for market regime detection
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        volume_ma_period = config['volume_ma']['period']
        df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        atr_period = config['atr']['period']
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        # ADX
        adx_period = config.get('adx', {}).get('period', 14)
        adx_threshold = config.get('adx', {}).get('threshold', 25)
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        minus_dm = -low_diff.where((low_diff > 0) & (low_diff > high_diff), 0)
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        smoothed_plus_dm = plus_dm.rolling(window=adx_period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=adx_period).sum()
        smoothed_tr = tr.rolling(window=adx_period).sum()
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=adx_period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        # Stochastic RSI
        stoch_rsi_period = config.get('stochastic_rsi', {}).get('rsi_period', 14)
        stoch_period = config.get('stochastic_rsi', {}).get('stochastic_period', 14)
        k_period = config.get('stochastic_rsi', {}).get('k_period', 3)
        d_period = config.get('stochastic_rsi', {}).get('d_period', 3)
        rsi_delta = df['close'].diff()
        rsi_gain = rsi_delta.where(rsi_delta > 0, 0)
        rsi_loss = -rsi_delta.where(rsi_delta < 0, 0)
        rsi_avg_gain = rsi_gain.rolling(window=stoch_rsi_period).mean()
        rsi_avg_loss = rsi_loss.rolling(window=stoch_rsi_period).mean()
        rsi_rs = rsi_avg_gain / rsi_avg_loss
        rsi = 100 - (100 / (1 + rsi_rs))
        stoch_rsi = (rsi - rsi.rolling(window=stoch_period).min()) / \
                    (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
        df['stoch_rsi_k'] = stoch_rsi.rolling(window=k_period).mean() * 100
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=d_period).mean()
        obv_smoothing = config.get('obv', {}).get('smoothing', 5)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(window=obv_smoothing).mean()
        return df

    @staticmethod
    def _calculate_signal_strength(data: np.ndarray) -> float:
        """Calculate signal strength using numpy arrays"""
        strength = 0.0
        
        # RSI signals
        rsi = data[0]  # Assuming RSI is the first column
        if rsi < 30:
            strength += 0.3
        elif rsi > 70:
            strength -= 0.3
        
        # MACD signals
        macd = data[1]  # Assuming MACD is the second column
        macd_signal = data[2]  # Assuming MACD signal is the third column
        if macd > macd_signal:
            strength += 0.2
        else:
            strength -= 0.2
        
        # Bollinger Bands signals
        close = data[3]  # Assuming close price is the fourth column
        bb_upper = data[4]  # Assuming BB upper is the fifth column
        bb_lower = data[5]  # Assuming BB lower is the sixth column
        if close < bb_lower:
            strength += 0.3
        elif close > bb_upper:
            strength -= 0.3
        
        return strength

    def _check_trading_schedule(self) -> bool:
        """Check if current time is within trading schedule"""
        try:
            schedule_config = self.config.get('schedule', {})
            trading_hours = schedule_config.get('trading_hours', {})
            trading_days = schedule_config.get('trading_days', [])
            market_events = schedule_config.get('market_events', {})
            
            # Get current time in UTC
            current_time = datetime.utcnow()
            current_day = current_time.strftime('%A')
            current_hour = current_time.strftime('%H:%M')
            
            # Check trading days
            if current_day not in trading_days:
                return False
            
            # Check trading hours
            start_time = trading_hours.get('start', '00:00')
            end_time = trading_hours.get('end', '23:59')
            if not (start_time <= current_hour <= end_time):
                return False
            
            # Check market events
            if market_events.get('avoid_news', True):
                # TODO: Implement news checking
                pass
            
            if market_events.get('avoid_high_impact', True):
                # TODO: Implement high impact event checking
                pass
            
            if market_events.get('avoid_volatility', False):
                # Check current volatility
                for symbol in self.data:
                    if symbol in self.data and '1h' in self.data[symbol]:
                        df = self.data[symbol]['1h']
                        if not df.empty:
                            current_volatility = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
                            if current_volatility > 0.1:  # High volatility threshold
                                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trading schedule: {str(e)}")
            return False

    def get_trading_signal(self, symbol: str) -> Dict:
        """Generate trading signal based on current market conditions."""
        try:
            # Check if we have recent data
            if not self._has_recent_data(symbol):
                return {'action': 'none', 'grade': 'F', 'reason': 'no_recent_data'}

            # Calculate indicators for all timeframes
            for timeframe in self.timeframes:
                if timeframe in self.data[symbol]:
                    df_with_indicators = self.calculate_indicators(symbol, timeframe)
                    if not df_with_indicators.empty:
                        self.data[symbol][timeframe] = df_with_indicators

            # Get latest data and indicators
            latest_data = self._get_latest_data(symbol)
            if not latest_data:
                return {'action': 'none', 'grade': 'F', 'reason': 'no_latest_data'}

            # Get market regime
            regime = self._detect_market_regime(symbol)
            if regime['regime'] == 'unknown' or regime['confidence'] < 0.6:
                return {'action': 'none', 'grade': 'F', 'reason': 'unclear_market_regime'}

            # Generate base signal
            signal = self._generate_base_signal(symbol, latest_data)
            if signal['action'] == 'none':
                return signal

            # Adjust signal based on market regime
            adjusted_signal = self._adjust_strategy_for_regime(signal, regime)

            # Log signal details
            self.logger.info(
                f"[SIGNAL] {symbol}: action={adjusted_signal['action']}, "
                f"grade={adjusted_signal['grade']}, strength={adjusted_signal['strength']}, "
                f"reasons={adjusted_signal['reasons']}, regime={regime['regime']}, "
                f"confidence={regime['confidence']}"
            )

            return adjusted_signal

        except Exception as e:
            self.logger.error(f"Error generating trading signal: {str(e)}", exc_info=True)
            return {'action': 'none', 'grade': 'F', 'reason': 'error'}

    def _has_recent_data(self, symbol: str) -> bool:
        """Check if we have recent data for all required timeframes."""
        try:
            current_time = datetime.now()
            for timeframe in self.timeframes:
                if timeframe not in self._last_signal_time[symbol]:
                    return False
                time_diff = current_time - self._last_signal_time[symbol][timeframe]
                if time_diff.total_seconds() > self._get_timeframe_seconds(timeframe) * 2:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking recent data: {str(e)}")
            return False

    def _get_latest_data(self, symbol: str) -> Dict:
        """Get latest data for all timeframes."""
        try:
            latest_data = {}
            for timeframe in self.timeframes:
                if timeframe in self.data[symbol] and not self.data[symbol][timeframe].empty:
                    latest_data[timeframe] = self.data[symbol][timeframe].iloc[-1]
            return latest_data
        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            return {}

    def _generate_base_signal(self, symbol: str, latest_data: Dict) -> Dict:
        """Generate base trading signal from indicators."""
        try:
            signal_strength = 0.0
            reasons = []
            confirmation_count = 0

            # RSI signals
            rsi = latest_data['1h']['rsi']
            if rsi < 25:
                signal_strength += 0.3
                reasons.append('oversold')
                confirmation_count += 1
            elif rsi > 75:
                signal_strength -= 0.3
                reasons.append('overbought')
                confirmation_count += 1

            # MACD signals
            macd = latest_data['1h']['macd']
            macd_signal = latest_data['1h']['macd_signal']
            macd_hist = latest_data['1h']['macd_hist']
            
            if macd > macd_signal and macd_hist > 0:
                signal_strength += 0.2
                reasons.append('macd_bullish')
                confirmation_count += 1
            elif macd < macd_signal and macd_hist < 0:
                signal_strength -= 0.2
                reasons.append('macd_bearish')
                confirmation_count += 1

            # Bollinger Bands signals
            bb_lower = latest_data['1h']['bb_lower']
            bb_upper = latest_data['1h']['bb_upper']
            close = latest_data['1h']['close']
            
            if close < bb_lower:
                signal_strength += 0.3
                reasons.append('bb_oversold')
                confirmation_count += 1
            elif close > bb_upper:
                signal_strength -= 0.3
                reasons.append('bb_overbought')
                confirmation_count += 1

            # EMA signals
            ema_short = latest_data['1h']['ema_short']
            ema_medium = latest_data['1h']['ema_medium']
            ema_long = latest_data['1h']['ema_long']
            
            if ema_short > ema_medium:
                signal_strength += 0.2
                reasons.append('ema_bullish')
                confirmation_count += 1
            elif ema_short < ema_medium:
                signal_strength -= 0.2
                reasons.append('ema_bearish')
                confirmation_count += 1

            # Volume confirmation
            volume_ratio = latest_data['1h']['volume'] / latest_data['1h']['volume_ma']
            if volume_ratio > 2.0:
                if signal_strength > 0:
                    signal_strength += 0.1
                    reasons.append('high_volume_bullish')
                    confirmation_count += 1
                elif signal_strength < 0:
                    signal_strength -= 0.1
                    reasons.append('high_volume_bearish')
                    confirmation_count += 1

            # Check minimum confirmations
            if confirmation_count < 3:
                return {'action': 'none', 'grade': 'F', 'reason': 'insufficient_confirmations'}

            # Determine signal grade
            if abs(signal_strength) >= 0.8:
                grade = 'AA+'
            elif abs(signal_strength) >= 0.6:
                grade = 'A'
            elif abs(signal_strength) >= 0.4:
                grade = 'B'
            elif abs(signal_strength) >= 0.2:
                grade = 'C'
            else:
                grade = 'F'

            # Check minimum grade
            if grade not in ['AA+', 'A', 'B']:
                return {'action': 'none', 'grade': grade, 'reason': 'low_confidence'}

            # Determine action
            action = 'buy' if signal_strength > 0 else 'sell' if signal_strength < 0 else 'none'

            return {
                'action': action,
                'grade': grade,
                'strength': abs(signal_strength),
                'reasons': reasons,
                'confidence': abs(signal_strength)
            }

        except Exception as e:
            self.logger.error(f"Error generating base signal: {str(e)}")
            return {'action': 'none', 'grade': 'F', 'reason': 'error'}

    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            if unit == 'm':
                return value * 60
            elif unit == 'h':
                return value * 3600
            elif unit == 'd':
                return value * 86400
            return 60  # Default to 1 minute
        except Exception as e:
            self.logger.error(f"Error converting timeframe: {str(e)}")
            return 60

    def should_exit_position(self, symbol: str, position: Dict) -> Tuple[bool, str, float]:
        """Determine if a position should be exited based on current market conditions."""
        try:
            latest_data = {}
            for timeframe in self.timeframes:
                if symbol not in self.data or timeframe not in self.data[symbol]:
                    return False, '', 0.0
                df = self.calculate_indicators(symbol, timeframe)
                if df.empty:
                    return False, '', 0.0
                latest_data[timeframe] = df.iloc[-1]

            strategy_config = self.config['strategy']
            exit_config = strategy_config['exit']
            current_price = latest_data['1m']['close']
            entry_price = position['entry_price']
            position_size = position['size']
            position_side = position['side']
            entry_time = position['entry_time']
            current_time = datetime.now()

            # Calculate PnL
            pnl = (current_price - entry_price) / entry_price if position_side == 'long' else (entry_price - current_price) / entry_price

            # Time-based exit
            time_in_trade = (current_time - entry_time).total_seconds() / 3600  # hours
            max_trade_duration = exit_config.get('max_trade_duration', 24)  # hours
            if time_in_trade > max_trade_duration:
                return True, 'time_exit', pnl

            # Trailing stop with dynamic activation
            trailing_cfg = exit_config.get('trailing_stop', {})
            if trailing_cfg.get('enabled', True):
                activation = trailing_cfg['activation']
                step = trailing_cfg['step']
                if pnl > activation:
                    # Dynamic trailing stop based on volatility
                    volatility = latest_data['1h']['atr'] / latest_data['1h']['close']
                    dynamic_step = max(step, volatility * 2)  # Adjust step based on volatility
                    trailing_stop = pnl - dynamic_step
                    if pnl < trailing_stop:
                        return True, 'trailing_stop', pnl

            # Multiple take-profit levels
            tp_cfg = exit_config.get('take_profit', {})
            if tp_cfg.get('enabled', True):
                levels = tp_cfg['levels']
                portions = tp_cfg['portions']
                for level, portion in zip(levels, portions):
                    if pnl >= level and position_size > portion:
                        return True, 'take_profit', pnl

            # Dynamic stop loss based on volatility
            sl_cfg = exit_config.get('stop_loss', {})
            if sl_cfg.get('enabled', True):
                initial_sl = sl_cfg['initial']
                breakeven = sl_cfg['breakeven']
                volatility = latest_data['1h']['atr'] / latest_data['1h']['close']
                
                # Adjust stop loss based on volatility
                dynamic_sl = max(initial_sl, volatility * 2)
                
                if pnl > breakeven:
                    # Move stop loss to breakeven plus small buffer
                    if pnl < breakeven + 0.001:  # 0.1% buffer
                        return True, 'breakeven_stop', pnl
                elif pnl < -dynamic_sl:
                    return True, 'stop_loss', pnl

            # Trend reversal exit
            trend_reversal = False
            ema_short = strategy_config['indicators']['ema']['short_period']
            ema_medium = strategy_config['indicators']['ema']['medium_period']
            
            # Check trend reversal on multiple timeframes
            for timeframe in ['1h', '4h']:
                if timeframe in latest_data:
                    data = latest_data[timeframe]
                    if position_side == 'long':
                        if (data[f'ema_{ema_short}'] < data[f'ema_{ema_medium}'] and 
                            data['macd'] < data['macd_signal'] and 
                            data['rsi'] > strategy_config['indicators']['rsi']['overbought']):
                            trend_reversal = True
                            break
                    else:
                        if (data[f'ema_{ema_short}'] > data[f'ema_{ema_medium}'] and 
                            data['macd'] > data['macd_signal'] and 
                            data['rsi'] < strategy_config['indicators']['rsi']['oversold']):
                            trend_reversal = True
                            break

            if trend_reversal:
                return True, 'trend_reversal', pnl

            # Volume-based exit
            for timeframe in ['1h', '4h']:
                if timeframe in latest_data:
                    data = latest_data[timeframe]
                    if data['volume_ratio'] > 3.0:  # High volume
                        if position_side == 'long' and data['close'] < data['open']:
                            return True, 'volume_exit', pnl
                        elif position_side == 'short' and data['close'] > data['open']:
                            return True, 'volume_exit', pnl

            return False, '', pnl

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {str(e)}", exc_info=True)
            return False, '', 0.0

    def calculate_position_size(self, symbol: str, price: float, signal: Dict) -> float:
        """Calculate position size based on risk management parameters."""
        try:
            # Get risk management parameters
            risk_config = self.config['risk']['position_sizing']
            symbol_config = self.config['symbols'][symbol]
            
            # Get account equity
            account_equity = self._get_account_equity()
            
            # Calculate base position size using configured risk
            base_risk = risk_config['base_risk_per_trade']
            max_position = risk_config['max_position_size']
            position_size = account_equity * base_risk / price
            
            # Adjust for signal confidence
            if signal and 'confidence' in signal:
                confidence = signal['confidence']
                position_size *= confidence
            
            # Apply symbol specific limits
            position_size = min(position_size, symbol_config['position_size'] * account_equity / price)
            
            # Apply portfolio limits
            position_size = min(position_size, max_position * account_equity / price)
            
            # Check correlation
            correlation = self._check_correlation(symbol)
            if correlation > risk_config['max_correlation']:
                position_size *= (1 - correlation)
            
            # Check volatility
            volatility = self._calculate_volatility(symbol)
            if volatility > 0.1:  # High volatility
                position_size *= 0.5
            elif volatility > 0.05:  # Medium volatility
                position_size *= 0.75
            
            # Get market regime and adjust position size
            regime = self._detect_market_regime(symbol)
            if regime['regime'] == 'trending':
                position_size *= 1.0  # Full size in trending markets
            elif regime['regime'] == 'ranging':
                position_size *= 0.7  # 70% size in ranging markets
            elif regime['regime'] == 'volatile':
                position_size *= 0.5  # 50% size in volatile markets
            elif regime['regime'] == 'low_volatility':
                position_size *= 0.8  # 80% size in low volatility
            
            # Check current drawdown and adjust position size
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > 0.05:  # If drawdown > 5%
                position_size *= (1 - current_drawdown)  # Reduce position size proportionally to drawdown
            
            # Ensure minimum position size
            min_position = account_equity * 0.01 / price  # 1% of account as minimum
            position_size = max(position_size, min_position)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}", exc_info=True)
            return 0.0

    def _get_daily_trades_count(self) -> int:
        """Get the number of trades executed today"""
        try:
            today = datetime.now().date()
            return sum(1 for trade in self.trades 
                      if trade['entry_time'].date() == today)
        except Exception as e:
            self.logger.error(f"Error getting daily trades count: {str(e)}")
            return 0

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            if not self.equity_curve:
                return 0.0
            
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            return (peak - current) / peak
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def _get_account_equity(self) -> float:
        """Get current account equity"""
        try:
            if not self.equity_curve:
                return self.initial_capital
            return self.equity_curve[-1]
        except Exception as e:
            self.logger.error(f"Error getting account equity: {str(e)}")
            return self.initial_capital

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current market volatility"""
        try:
            if symbol not in self.data or '1m' not in self.data[symbol]:
                return 1.0
            
            df = self.data[symbol]['1m']
            if df.empty:
                return 1.0
            
            # Calculate volatility using ATR
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            return atr / price
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 1.0

    def _check_correlation(self, symbol: str) -> float:
        """Check correlation with existing positions"""
        try:
            if not self.positions:
                return 0.0
            
            if symbol not in self.data or '1h' not in self.data[symbol]:
                return 0.0
            
            # Get price data for correlation calculation
            symbol_data = self.data[symbol]['1h']['close']
            
            max_correlation = 0.0
            for pos_symbol in self.positions:
                if pos_symbol not in self.data or '1h' not in self.data[pos_symbol]:
                    continue
                
                pos_data = self.data[pos_symbol]['1h']['close']
                correlation = symbol_data.corr(pos_data)
                max_correlation = max(max_correlation, abs(correlation))
            
            return max_correlation
        except Exception as e:
            self.logger.error(f"Error checking correlation: {str(e)}")
            return 0.0

    def _detect_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Detect current market regime based on various indicators"""
        try:
            if symbol not in self.data or '1h' not in self.data[symbol]:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            df = self.calculate_indicators(symbol, '1h')
            if df.empty:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            # Get latest values
            current = df.iloc[-1]
            
            # Initialize regime scores
            regimes = {
                'trending': 0.0,
                'ranging': 0.0,
                'volatile': 0.0,
                'low_volatility': 0.0
            }
            
            # Check for trending market
            adx = current['adx']
            ema_20 = current['ema_20']
            ema_50 = current['ema_50']
            ema_200 = current['ema_200']
            
            if adx > 25:  # Strong trend
                regimes['trending'] += 0.4
                if ema_20 > ema_50 > ema_200:  # Uptrend
                    regimes['trending'] += 0.3
                elif ema_20 < ema_50 < ema_200:  # Downtrend
                    regimes['trending'] += 0.3
                
            # Check for ranging market
            bb_width = (current['bb_upper'] - current['bb_lower']) / current['bb_middle']
            if bb_width < 0.05:  # Narrow Bollinger Bands
                regimes['ranging'] += 0.4
            if adx < 20:  # Weak trend
                regimes['ranging'] += 0.3
            
            # Check for volatile market
            atr = current['atr']
            atr_percent = atr / current['close']
            if atr_percent > 0.02:  # High volatility
                regimes['volatile'] += 0.4
            if bb_width > 0.1:  # Wide Bollinger Bands
                regimes['volatile'] += 0.3
            
            # Check for low volatility
            if atr_percent < 0.005:  # Low volatility
                regimes['low_volatility'] += 0.4
            if bb_width < 0.02:  # Very narrow Bollinger Bands
                regimes['low_volatility'] += 0.3
            
            # Determine dominant regime
            dominant_regime = max(regimes.items(), key=lambda x: x[1])
            
            return {
                'regime': dominant_regime[0],
                'confidence': dominant_regime[1],
                'scores': regimes
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.0}

    def _adjust_strategy_for_regime(self, signal: Dict, regime: Dict) -> Dict:
        """Adjust trading strategy parameters based on market regime"""
        try:
            adjusted_signal = signal.copy()
            
            # Get regime-specific parameters
            regime_config = self.config.get('market_regime', {})
            regime_params = regime_config.get(regime['regime'], {})
            
            # Adjust signal confidence based on regime
            regime_confidence = regime['confidence']
            adjusted_signal['confidence'] *= regime_confidence
            
            # Adjust position sizing based on regime
            if regime['regime'] == 'trending':
                adjusted_signal['position_multiplier'] = regime_params.get('position_multiplier', 1.2)
            elif regime['regime'] == 'ranging':
                adjusted_signal['position_multiplier'] = regime_params.get('position_multiplier', 0.8)
            elif regime['regime'] == 'volatile':
                adjusted_signal['position_multiplier'] = regime_params.get('position_multiplier', 0.6)
            elif regime['regime'] == 'low_volatility':
                adjusted_signal['position_multiplier'] = regime_params.get('position_multiplier', 1.0)
            
            # Adjust stop loss and take profit levels
            if regime['regime'] == 'volatile':
                adjusted_signal['stop_loss_multiplier'] = regime_params.get('stop_loss_multiplier', 1.5)
                adjusted_signal['take_profit_multiplier'] = regime_params.get('take_profit_multiplier', 2.0)
            else:
                adjusted_signal['stop_loss_multiplier'] = regime_params.get('stop_loss_multiplier', 1.0)
                adjusted_signal['take_profit_multiplier'] = regime_params.get('take_profit_multiplier', 1.0)
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Error adjusting strategy for regime: {str(e)}")
            return signal 

    def close_all_positions(self, reason: str = 'max_drawdown'):
        """Close all open positions immediately, e.g. when max drawdown is hit."""
        closed = []
        for symbol, pos in list(self.positions.items()):
            # Here, you would implement the actual closing logic. For now, just log and remove.
            self.logger.info(f"[CLOSE ALL] Closing position for {symbol} due to {reason}.")
            closed.append(symbol)
            del self.positions[symbol]
        return closed 

    def step(self, symbol: str):
        """Main step for the strategy, to be called each bar/tick."""
        # Check drawdown and close all positions if needed
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = self.config['risk']['position_sizing']['max_drawdown']
        if current_drawdown >= max_drawdown:
            self.logger.warning(f"Max drawdown limit reached: {current_drawdown*100:.1f}% >= {max_drawdown*100:.1f}%")
            self.close_all_positions(reason='max_drawdown')
            return  # Skip further processing when max drawdown is hit
        # ... rest of your trading logic ... 

    def set_equity_curve(self, equity_curve):
        self.equity_curve = equity_curve 

    def update_trade_outcome(self, symbol: str, trade_data: Dict, outcome: Dict):
        """Update the AI module with trade outcomes"""
        try:
            # Get market data
            market_data = {
                'open': self.data[symbol]['1h']['open'],
                'high': self.data[symbol]['1h']['high'],
                'low': self.data[symbol]['1h']['low'],
                'close': self.data[symbol]['1h']['close'],
                'volume': self.data[symbol]['1h']['volume'],
                **self.indicators[symbol]['1h']
            }
            
            # Update AI module
            self.ai_module.update(trade_data, market_data, outcome)
            
        except Exception as e:
            self.logger.error(f"Error updating trade outcome: {str(e)}")
            
    def save_state(self):
        """Save strategy and AI state"""
        try:
            # Save AI module state
            self.ai_module.save_state()
            
            # Save strategy state
            # TODO: Implement strategy state saving
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            
    def load_state(self):
        """Load strategy and AI state"""
        try:
            # Load AI module state
            self.ai_module.load_state()
            
            # Load strategy state
            # TODO: Implement strategy state loading
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}") 