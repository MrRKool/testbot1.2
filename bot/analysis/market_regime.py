import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class MarketRegime:
    name: str
    volatility: float
    trend_strength: float
    volume_profile: float
    probability: float

class MarketRegimeDetector:
    """Detects different market regimes using various technical indicators."""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        self.regime_models = {
            'trending': self._detect_trending,
            'ranging': self._detect_ranging,
            'volatile': self._detect_volatile,
            'reversal': self._detect_reversal
        }
        
    def detect_regime(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Detect current market regime and return probabilities for each regime."""
        try:
            # Calculate base indicators
            volatility = self._calculate_volatility(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            volume_profile = self._calculate_volume_profile(market_data)
            
            # Get regime probabilities
            regime_probs = {}
            for regime_name, detector in self.regime_models.items():
                prob = detector(market_data, volatility, trend_strength, volume_profile)
                regime_probs[regime_name] = prob
                
            # Normalize probabilities
            total_prob = sum(regime_probs.values())
            if total_prob > 0:
                regime_probs = {k: v/total_prob for k, v in regime_probs.items()}
                
            return regime_probs
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return {}
            
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility."""
        try:
            returns = data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)  # Annualized volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX."""
        try:
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = data['high'] - data['high'].shift()
            down_move = data['low'].shift() - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed averages
            tr_smooth = tr.rolling(14).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / tr_smooth
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / tr_smooth
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            
            return adx.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
            
    def _calculate_volume_profile(self, data: pd.DataFrame) -> float:
        """Calculate volume profile and its trend."""
        try:
            # Calculate volume moving average
            volume_ma = data['volume'].rolling(20).mean()
            
            # Calculate volume trend
            volume_trend = (data['volume'] / volume_ma - 1).rolling(5).mean()
            
            return volume_trend.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return 0.0
            
    def _detect_trending(self, data: pd.DataFrame, volatility: float, 
                        trend_strength: float, volume_profile: float) -> float:
        """Detect trending market conditions."""
        try:
            # Strong trend indicators
            trend_prob = 0.0
            
            # Check ADX
            if trend_strength > 25:
                trend_prob += 0.4
                
            # Check price action
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            
            if (data['close'].iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1] or
                data['close'].iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]):
                trend_prob += 0.3
                
            # Check volume confirmation
            if volume_profile > 0.1:
                trend_prob += 0.3
                
            return trend_prob
            
        except Exception as e:
            self.logger.error(f"Error in trending detection: {e}")
            return 0.0
            
    def _detect_ranging(self, data: pd.DataFrame, volatility: float,
                       trend_strength: float, volume_profile: float) -> float:
        """Detect ranging market conditions."""
        try:
            range_prob = 0.0
            
            # Check ADX
            if trend_strength < 20:
                range_prob += 0.4
                
            # Check price action
            atr = self._calculate_atr(data)
            price_range = (data['high'].iloc[-20:] - data['low'].iloc[-20:]).mean()
            
            if price_range < 2 * atr:
                range_prob += 0.3
                
            # Check volume profile
            if abs(volume_profile) < 0.1:
                range_prob += 0.3
                
            return range_prob
            
        except Exception as e:
            self.logger.error(f"Error in ranging detection: {e}")
            return 0.0
            
    def _detect_volatile(self, data: pd.DataFrame, volatility: float,
                        trend_strength: float, volume_profile: float) -> float:
        """Detect volatile market conditions."""
        try:
            volatile_prob = 0.0
            
            # Check volatility
            if volatility > 0.5:  # High volatility threshold
                volatile_prob += 0.4
                
            # Check price action
            price_changes = data['close'].pct_change().abs()
            if price_changes.iloc[-5:].mean() > 0.02:  # 2% average daily change
                volatile_prob += 0.3
                
            # Check volume
            if volume_profile > 0.2:
                volatile_prob += 0.3
                
            return volatile_prob
            
        except Exception as e:
            self.logger.error(f"Error in volatile detection: {e}")
            return 0.0
            
    def _detect_reversal(self, data: pd.DataFrame, volatility: float,
                        trend_strength: float, volume_profile: float) -> float:
        """Detect potential market reversal conditions."""
        try:
            reversal_prob = 0.0
            
            # Check for divergence
            rsi = self._calculate_rsi(data)
            if self._check_divergence(data['close'], rsi):
                reversal_prob += 0.4
                
            # Check for overbought/oversold
            if rsi.iloc[-1] > 70 or rsi.iloc[-1] < 30:
                reversal_prob += 0.3
                
            # Check volume confirmation
            if volume_profile > 0.3:
                reversal_prob += 0.3
                
            return reversal_prob
            
        except Exception as e:
            self.logger.error(f"Error in reversal detection: {e}")
            return 0.0
            
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
            
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=data.index)
            
    def _check_divergence(self, price: pd.Series, indicator: pd.Series) -> bool:
        """Check for price-indicator divergence."""
        try:
            # Get recent highs and lows
            price_highs = price.rolling(5, center=True).max()
            indicator_highs = indicator.rolling(5, center=True).max()
            
            # Check for bearish divergence
            if (price_highs.iloc[-1] > price_highs.iloc[-2] and
                indicator_highs.iloc[-1] < indicator_highs.iloc[-2]):
                return True
                
            # Check for bullish divergence
            if (price_highs.iloc[-1] < price_highs.iloc[-2] and
                indicator_highs.iloc[-1] > indicator_highs.iloc[-2]):
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking divergence: {e}")
            return False 