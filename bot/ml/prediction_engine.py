import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

class MLPredictionEngine:
    """Machine Learning engine for market predictions."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.models = {
            'price': self._create_price_model(),
            'volatility': self._create_volatility_model(),
            'regime': self._create_regime_model()
        }
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower',
            'atr', 'adx', 'obv'
        ]
        
    def _create_price_model(self) -> tf.keras.Model:
        """Create LSTM model for price prediction."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(60, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _create_volatility_model(self) -> RandomForestRegressor:
        """Create Random Forest model for volatility prediction."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def _create_regime_model(self) -> GradientBoostingRegressor:
        """Create Gradient Boosting model for regime prediction."""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        try:
            # Calculate technical indicators
            data = self._add_technical_indicators(data)
            
            # Select features
            features = data[self.feature_columns].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.array([])
            
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = tr.rolling(14).mean()
            
            # ADX
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
            minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            data['adx'] = dx.rolling(14).mean()
            
            # OBV
            data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data
            
    async def train_models(self, data: pd.DataFrame):
        """Train all prediction models."""
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Prepare targets
            price_target = data['close'].shift(-1).dropna().values
            volatility_target = data['close'].pct_change().rolling(20).std().dropna().values
            regime_target = self._prepare_regime_target(data)
            
            # Train price model
            X_price, y_price = self._prepare_sequences(features, price_target)
            self.models['price'].fit(
                X_price, y_price,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Train volatility model
            X_vol, y_vol = train_test_split(
                features[:-1], volatility_target,
                test_size=0.2,
                random_state=42
            )
            self.models['volatility'].fit(X_vol, y_vol)
            
            # Train regime model
            X_regime, y_regime = train_test_split(
                features[:-1], regime_target,
                test_size=0.2,
                random_state=42
            )
            self.models['regime'].fit(X_regime, y_regime)
            
            # Save models
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            
    def _prepare_sequences(self, features: np.ndarray, target: np.ndarray,
                          sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model."""
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
        
    def _prepare_regime_target(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare regime classification target."""
        # Simple regime classification based on price action
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        regime = np.zeros(len(data))
        regime[returns > 0.001] = 1  # Uptrend
        regime[returns < -0.001] = 2  # Downtrend
        regime[volatility > 0.02] = 3  # High volatility
        
        return regime[:-1]  # Remove last row as it's used for prediction
        
    async def predict(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions for all models."""
        try:
            features = self.prepare_features(data)
            
            # Price prediction
            X_price = self._prepare_sequences(features, np.zeros(len(features)))[0]
            price_pred = self.models['price'].predict(X_price[-1:])[0][0]
            
            # Volatility prediction
            volatility_pred = self.models['volatility'].predict(features[-1:])[0]
            
            # Regime prediction
            regime_pred = self.models['regime'].predict(features[-1:])[0]
            
            return {
                'price': price_pred,
                'volatility': volatility_pred,
                'regime': regime_pred
            }
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {}
            
    def _save_models(self):
        """Save trained models to disk."""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save LSTM model
            self.models['price'].save(f"{self.model_dir}/price_model.h5")
            
            # Save other models
            joblib.dump(self.models['volatility'], f"{self.model_dir}/volatility_model.joblib")
            joblib.dump(self.models['regime'], f"{self.model_dir}/regime_model.joblib")
            
            # Save scaler
            joblib.dump(self.scaler, f"{self.model_dir}/scaler.joblib")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            
    def load_models(self):
        """Load trained models from disk."""
        try:
            # Load LSTM model
            self.models['price'] = tf.keras.models.load_model(f"{self.model_dir}/price_model.h5")
            
            # Load other models
            self.models['volatility'] = joblib.load(f"{self.model_dir}/volatility_model.joblib")
            self.models['regime'] = joblib.load(f"{self.model_dir}/regime_model.joblib")
            
            # Load scaler
            self.scaler = joblib.load(f"{self.model_dir}/scaler.joblib")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        try:
            features = self.prepare_features(test_data)
            
            # Prepare test targets
            price_target = test_data['close'].shift(-1).dropna().values
            volatility_target = test_data['close'].pct_change().rolling(20).std().dropna().values
            regime_target = self._prepare_regime_target(test_data)
            
            # Evaluate price model
            X_price, y_price = self._prepare_sequences(features, price_target)
            price_score = self.models['price'].evaluate(X_price, y_price, verbose=0)
            
            # Evaluate volatility model
            X_vol, y_vol = train_test_split(
                features[:-1], volatility_target,
                test_size=0.2,
                random_state=42
            )
            volatility_score = self.models['volatility'].score(X_vol, y_vol)
            
            # Evaluate regime model
            X_regime, y_regime = train_test_split(
                features[:-1], regime_target,
                test_size=0.2,
                random_state=42
            )
            regime_score = self.models['regime'].score(X_regime, y_regime)
            
            return {
                'price_mse': price_score,
                'volatility_r2': volatility_score,
                'regime_accuracy': regime_score
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            return {} 