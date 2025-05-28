from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"

@dataclass
class Signal:
    type: SignalType
    strength: float
    metadata: Dict

class ExampleStrategy:
    """Example trading strategy using technical indicators."""
    
    def __init__(self, config: Dict):
        """Initialize strategy with configuration."""
        self.config = config
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        
        self.sma_fast = config.get('sma_fast', 20)
        self.sma_slow = config.get('sma_slow', 50)
        
    def generate_signals(self, data: pd.Series) -> List[Signal]:
        """Generate trading signals based on technical indicators."""
        signals = []
        
        try:
            # 1. RSI signals
            rsi_signal = self._check_rsi(data)
            if rsi_signal:
                signals.append(rsi_signal)
                
            # 2. MACD signals
            macd_signal = self._check_macd(data)
            if macd_signal:
                signals.append(macd_signal)
                
            # 3. Bollinger Bands signals
            bb_signal = self._check_bollinger_bands(data)
            if bb_signal:
                signals.append(bb_signal)
                
            # 4. Moving Average signals
            ma_signal = self._check_moving_averages(data)
            if ma_signal:
                signals.append(ma_signal)
                
            # Combine signals
            if signals:
                return self._combine_signals(signals)
                
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            
        return signals
        
    def _check_rsi(self, data: pd.Series) -> Optional[Signal]:
        """Check RSI for trading signals."""
        try:
            rsi = data['RSI']
            
            if rsi <= self.rsi_oversold:
                return Signal(
                    type=SignalType.LONG,
                    strength=1.0 - (rsi / self.rsi_oversold),
                    metadata={'indicator': 'RSI', 'value': rsi}
                )
            elif rsi >= self.rsi_overbought:
                return Signal(
                    type=SignalType.SHORT,
                    strength=1.0 - ((100 - rsi) / (100 - self.rsi_overbought)),
                    metadata={'indicator': 'RSI', 'value': rsi}
                )
                
        except Exception as e:
            print(f"Error checking RSI: {str(e)}")
            
        return None
        
    def _check_macd(self, data: pd.Series) -> Optional[Signal]:
        """Check MACD for trading signals."""
        try:
            macd = data['MACD']
            signal = data['Signal']
            
            if macd > signal:
                return Signal(
                    type=SignalType.LONG,
                    strength=min(1.0, (macd - signal) / abs(signal)),
                    metadata={'indicator': 'MACD', 'value': macd, 'signal': signal}
                )
            elif macd < signal:
                return Signal(
                    type=SignalType.SHORT,
                    strength=min(1.0, (signal - macd) / abs(signal)),
                    metadata={'indicator': 'MACD', 'value': macd, 'signal': signal}
                )
                
        except Exception as e:
            print(f"Error checking MACD: {str(e)}")
            
        return None
        
    def _check_bollinger_bands(self, data: pd.Series) -> Optional[Signal]:
        """Check Bollinger Bands for trading signals."""
        try:
            price = data['close']
            upper = data['BB_upper']
            lower = data['BB_lower']
            middle = data['BB_middle']
            
            if price <= lower:
                return Signal(
                    type=SignalType.LONG,
                    strength=min(1.0, (middle - price) / (middle - lower)),
                    metadata={'indicator': 'BB', 'value': price, 'lower': lower}
                )
            elif price >= upper:
                return Signal(
                    type=SignalType.SHORT,
                    strength=min(1.0, (price - middle) / (upper - middle)),
                    metadata={'indicator': 'BB', 'value': price, 'upper': upper}
                )
                
        except Exception as e:
            print(f"Error checking Bollinger Bands: {str(e)}")
            
        return None
        
    def _check_moving_averages(self, data: pd.Series) -> Optional[Signal]:
        """Check Moving Averages for trading signals."""
        try:
            sma_fast = data['SMA_20']
            sma_slow = data['SMA_50']
            
            if sma_fast > sma_slow:
                return Signal(
                    type=SignalType.LONG,
                    strength=min(1.0, (sma_fast - sma_slow) / sma_slow),
                    metadata={'indicator': 'SMA', 'fast': sma_fast, 'slow': sma_slow}
                )
            elif sma_fast < sma_slow:
                return Signal(
                    type=SignalType.SHORT,
                    strength=min(1.0, (sma_slow - sma_fast) / sma_slow),
                    metadata={'indicator': 'SMA', 'fast': sma_fast, 'slow': sma_slow}
                )
                
        except Exception as e:
            print(f"Error checking Moving Averages: {str(e)}")
            
        return None
        
    def _combine_signals(self, signals: List[Signal]) -> List[Signal]:
        """Combine multiple signals into final trading decisions."""
        try:
            # Count signal types
            long_signals = [s for s in signals if s.type == SignalType.LONG]
            short_signals = [s for s in signals if s.type == SignalType.SHORT]
            
            # Calculate average strengths
            long_strength = np.mean([s.strength for s in long_signals]) if long_signals else 0
            short_strength = np.mean([s.strength for s in short_signals]) if short_signals else 0
            
            # Generate final signals
            final_signals = []
            
            if long_strength > 0.5:  # Strong long signal
                final_signals.append(Signal(
                    type=SignalType.LONG,
                    strength=long_strength,
                    metadata={'combined': True, 'signals': len(long_signals)}
                ))
            elif short_strength > 0.5:  # Strong short signal
                final_signals.append(Signal(
                    type=SignalType.SHORT,
                    strength=short_strength,
                    metadata={'combined': True, 'signals': len(short_signals)}
                ))
                
            return final_signals
            
        except Exception as e:
            print(f"Error combining signals: {str(e)}")
            return [] 