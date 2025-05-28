import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

class MarketAnalyzer:
    """Analyzes market conditions and detects market regimes."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize market regime model
        self.regime_model = KMeans(n_clusters=3, random_state=42)
        
        # Create cache directory
        self.cache_dir = "cache/market"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load or train models
        self._load_or_train_models()
        
    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        try:
            # Try to load scaler
            scaler_path = os.path.join(self.cache_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            # Try to load regime model
            regime_path = os.path.join(self.cache_dir, "regime_model.joblib")
            if os.path.exists(regime_path):
                self.regime_model = joblib.load(regime_path)
                
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            
    def _save_models(self):
        """Save trained models."""
        try:
            # Save scaler
            scaler_path = os.path.join(self.cache_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Save regime model
            regime_path = os.path.join(self.cache_dir, "regime_model.joblib")
            joblib.dump(self.regime_model, regime_path)
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Analyze market conditions and detect regime."""
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Detect market regime
            regime = self._detect_regime(indicators)
            
            # Calculate market conditions
            conditions = self._calculate_market_conditions(data, indicators)
            
            # Combine results
            analysis = {
                'regime': regime,
                'conditions': conditions,
                'indicators': indicators
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {
                'regime': 'unknown',
                'conditions': {},
                'indicators': {}
            }
            
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        try:
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required price data columns")
                
            # Calculate basic indicators
            indicators = {}
            
            # Trend indicators
            indicators['sma_20'] = data.ta.sma(length=20)
            indicators['sma_50'] = data.ta.sma(length=50)
            indicators['sma_200'] = data.ta.sma(length=200)
            indicators['ema_20'] = data.ta.ema(length=20)
            
            # Momentum indicators
            indicators['rsi'] = data.ta.rsi(length=14)
            macd = data.ta.macd()
            indicators['macd'] = macd['MACD_12_26_9']
            indicators['macd_signal'] = macd['MACDs_12_26_9']
            indicators['macd_hist'] = macd['MACDh_12_26_9']
            
            # Volatility indicators
            indicators['atr'] = data.ta.atr(length=14)
            bbands = data.ta.bbands(length=20, std=2)
            indicators['bbands_upper'] = bbands['BBU_20_2.0']
            indicators['bbands_middle'] = bbands['BBM_20_2.0']
            indicators['bbands_lower'] = bbands['BBL_20_2.0']
            
            # Volume indicators
            indicators['obv'] = data.ta.obv()
            indicators['ad'] = data.ta.ad()
            
            # Custom indicators
            indicators['volatility'] = self._calculate_volatility(data)
            indicators['trend_strength'] = self._calculate_trend_strength(data)
            indicators['volume_profile'] = self._calculate_volume_profile(data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
            
    def _detect_regime(self, indicators: Dict) -> str:
        """Detect market regime using clustering."""
        try:
            # Prepare features for regime detection
            features = pd.DataFrame({
                'trend': indicators['trend_strength'],
                'volatility': indicators['volatility'],
                'volume': indicators['volume_profile'],
                'momentum': indicators['rsi']
            }).dropna()
            
            if features.empty:
                return 'unknown'
                
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Predict regime
            labels = self.regime_model.fit_predict(scaled_features)
            
            # Map labels to regimes
            regime_map = {
                0: 'ranging',
                1: 'trending',
                2: 'volatile'
            }
            
            # Get most recent regime
            current_regime = regime_map[labels[-1]]
            
            # Save updated models
            self._save_models()
            
            return current_regime
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {str(e)}")
            return 'unknown'
            
    def _calculate_market_conditions(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Calculate current market conditions."""
        try:
            conditions = {}
            
            # Trend conditions
            conditions['trend'] = self._analyze_trend(data, indicators)
            
            # Volatility conditions
            conditions['volatility'] = self._analyze_volatility(data, indicators)
            
            # Volume conditions
            conditions['volume'] = self._analyze_volume(data, indicators)
            
            # Support/Resistance levels
            conditions['levels'] = self._find_support_resistance(data)
            
            # Market structure
            conditions['structure'] = self._analyze_market_structure(data)
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error calculating market conditions: {str(e)}")
            return {}
            
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate price volatility."""
        returns = data['close'].pct_change()
        return returns.rolling(window=window).std()
        
    def _calculate_trend_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate trend strength using ADX."""
        adx = data.ta.adx(length=window)
        return adx[f'ADX_{window}']
        
    def _calculate_volume_profile(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate volume profile."""
        return data['volume'] / data['volume'].rolling(window=window).mean()
        
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze current trend conditions."""
        try:
            current_price = data['close'].iloc[-1]
            sma_20 = indicators['sma_20'].iloc[-1]
            sma_50 = indicators['sma_50'].iloc[-1]
            sma_200 = indicators['sma_200'].iloc[-1]
            
            # Determine trend direction
            if current_price > sma_20 > sma_50 > sma_200:
                trend = 'strong_uptrend'
            elif current_price < sma_20 < sma_50 < sma_200:
                trend = 'strong_downtrend'
            elif current_price > sma_20 and sma_20 > sma_50:
                trend = 'uptrend'
            elif current_price < sma_20 and sma_20 < sma_50:
                trend = 'downtrend'
            else:
                trend = 'sideways'
                
            # Calculate trend strength
            strength = indicators['trend_strength'].iloc[-1]
            
            return {
                'direction': trend,
                'strength': strength,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return {
                'direction': 'unknown',
                'strength': 0.0,
                'sma_20': 0.0,
                'sma_50': 0.0,
                'sma_200': 0.0
            }
            
    def _analyze_volatility(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze current volatility conditions."""
        try:
            current_atr = indicators['atr'].iloc[-1]
            current_bb_width = (indicators['bbands_upper'].iloc[-1] - indicators['bbands_lower'].iloc[-1]) / indicators['bbands_middle'].iloc[-1]
            
            # Determine volatility regime
            if current_bb_width > 0.1:  # High volatility
                regime = 'high'
            elif current_bb_width < 0.05:  # Low volatility
                regime = 'low'
            else:
                regime = 'normal'
                
            return {
                'regime': regime,
                'atr': current_atr,
                'bb_width': current_bb_width
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            return {
                'regime': 'unknown',
                'atr': 0.0,
                'bb_width': 0.0
            }
            
    def _analyze_volume(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze current volume conditions."""
        try:
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            obv = indicators['obv'].iloc[-1]
            
            # Determine volume regime
            if current_volume > avg_volume * 1.5:
                regime = 'high'
            elif current_volume < avg_volume * 0.5:
                regime = 'low'
            else:
                regime = 'normal'
                
            return {
                'regime': regime,
                'current': current_volume,
                'average': avg_volume,
                'obv': obv
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            return {
                'regime': 'unknown',
                'current': 0.0,
                'average': 0.0,
                'obv': 0.0
            }
            
    def _find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Find support and resistance levels."""
        try:
            # Calculate pivot points
            pivots = self._find_pivot_points(data, window)
            
            # Group pivot points
            support_levels = []
            resistance_levels = []
            
            for i in range(1, len(pivots) - 1):
                if pivots[i] < pivots[i-1] and pivots[i] < pivots[i+1]:
                    support_levels.append(pivots[i])
                elif pivots[i] > pivots[i-1] and pivots[i] > pivots[i+1]:
                    resistance_levels.append(pivots[i])
                    
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Find nearest levels
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {str(e)}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None
            }
            
    def _find_pivot_points(self, data: pd.DataFrame, window: int) -> List[float]:
        """Find pivot points in price data."""
        try:
            pivots = []
            for i in range(window, len(data) - window):
                if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, window+1)):
                    pivots.append(data['low'].iloc[i])
                elif all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, window+1)) and \
                     all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, window+1)):
                    pivots.append(data['high'].iloc[i])
            return pivots
            
        except Exception as e:
            self.logger.error(f"Error finding pivot points: {str(e)}")
            return []
            
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure (higher highs, lower lows, etc.)."""
        try:
            # Find recent highs and lows
            highs = self._find_pivot_points(data[['high']], window=5)
            lows = self._find_pivot_points(data[['low']], window=5)
            
            # Analyze structure
            if len(highs) >= 2 and len(lows) >= 2:
                if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                    structure = 'higher_highs_higher_lows'
                elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                    structure = 'lower_highs_lower_lows'
                elif highs[-1] > highs[-2] and lows[-1] < lows[-2]:
                    structure = 'higher_highs_lower_lows'
                else:
                    structure = 'lower_highs_higher_lows'
            else:
                structure = 'undefined'
                
            return {
                'structure': structure,
                'recent_highs': highs[-2:] if len(highs) >= 2 else [],
                'recent_lows': lows[-2:] if len(lows) >= 2 else []
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {str(e)}")
            return {
                'structure': 'unknown',
                'recent_highs': [],
                'recent_lows': []
            } 