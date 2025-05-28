import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all AI models in the trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.performance_metrics = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self):
        """Initialize the model with configuration."""
        raise NotImplementedError("Subclasses must implement initialize()")
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        raise NotImplementedError("Subclasses must implement prepare_features()")
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        raise NotImplementedError("Subclasses must implement train()")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the model."""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError("Subclasses must implement evaluate()")
        
    def save(self, path: str):
        """Save the model to disk."""
        raise NotImplementedError("Subclasses must implement save()")
        
    def load(self, path: str):
        """Load the model from disk."""
        raise NotImplementedError("Subclasses must implement load()")
        
    def _create_keras_predict_fn(self, model):
        """Create a TensorFlow function for prediction."""
        @tf.function(reduce_retracing=True)
        def keras_predict(X):
            return model(X, training=False)
        return keras_predict
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        try:
            # RSI
            data['rsi'] = ta.rsi(data['close'], length=14)
            
            # MACD
            macd = ta.macd(data['close'])
            data = pd.concat([data, macd], axis=1)
            
            # Bollinger Bands
            bb = ta.bbands(data['close'])
            data = pd.concat([data, bb], axis=1)
            
            # EMAs
            data['ema_short'] = ta.ema(data['close'], length=9)
            data['ema_medium'] = ta.ema(data['close'], length=21)
            data['ema_long'] = ta.ema(data['close'], length=50)
            
            # Volume MA
            data['volume_ma'] = ta.sma(data['volume'], length=20)
            
            # ATR
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
            
    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features."""
        try:
            # Price impact
            data['price_impact'] = (data['high'] - data['low']) / data['volume']
            
            # Volume profile
            data['volume_profile'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Spread
            data['spread'] = (data['high'] - data['low']) / data['close']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating microstructure features: {str(e)}")
            return data
            
    def _calculate_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment features."""
        try:
            # Price momentum
            data['price_momentum'] = data['close'].pct_change(5)
            
            # Volume momentum
            data['volume_momentum'] = data['volume'].pct_change(5)
            
            # Trend strength
            data['trend_strength'] = abs(data['ema_short'] - data['ema_long']) / data['ema_long']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment features: {str(e)}")
            return data
            
    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features."""
        try:
            # Historical volatility
            data['historical_vol'] = data['close'].pct_change().rolling(20).std()
            
            # Volatility ratio
            data['volatility_ratio'] = data['atr'] / data['close']
            
            # Volatility trend
            data['volatility_trend'] = data['historical_vol'].pct_change(5)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return data
            
    def _combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine different feature sets into a single array."""
        try:
            combined = []
            for feature_set in features.values():
                if isinstance(feature_set, np.ndarray):
                    combined.append(feature_set)
                elif isinstance(feature_set, pd.DataFrame):
                    combined.append(feature_set.values)
                else:
                    self.logger.warning(f"Unknown feature type: {type(feature_set)}")
                    
            return np.concatenate(combined, axis=1)
            
        except Exception as e:
            self.logger.error(f"Error combining features: {str(e)}")
            return np.array([]) 