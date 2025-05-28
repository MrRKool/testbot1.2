import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

class OnlineLearning:
    """Online learning implementatie voor real-time model updates."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.model_dir = config.get('model_dir', 'models')
        self.model_type = config.get('model_type', 'sgd')
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        
    def initialize_model(self, model_name: str, input_dim: int, output_dim: int = 1) -> bool:
        """Initialize a new model for online learning."""
        try:
            if self.model_type == 'sgd':
                # Initialize SGD model
                if output_dim == 1:
                    self.models[model_name] = SGDClassifier(
                        learning_rate='constant',
                        eta0=self.learning_rate,
                        random_state=42
                    )
                else:
                    self.models[model_name] = SGDRegressor(
                        learning_rate='constant',
                        eta0=self.learning_rate,
                        random_state=42
                    )
                    
            elif self.model_type == 'neural_network':
                # Initialize neural network
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.models[model_name] = model
                
            # Initialize scaler
            self.scalers[model_name] = StandardScaler()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            return False
            
    def update_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> bool:
        """Update model with new data."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
                
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)
            
            # Update model
            if self.model_type == 'sgd':
                self.models[model_name].partial_fit(X_scaled, y)
            else:
                self.models[model_name].fit(
                    X_scaled, y,
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=0
                )
                
            # Update metrics
            self._update_metrics(model_name, X_scaled, y)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False
            
    def _update_metrics(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Update model metrics."""
        try:
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = {
                    'updates': 0,
                    'last_update': None,
                    'accuracy': []
                }
                
            # Make predictions
            y_pred = self.predict(model_name, X)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y)
            
            # Update metrics
            self.model_metrics[model_name]['updates'] += 1
            self.model_metrics[model_name]['last_update'] = pd.Timestamp.now()
            self.model_metrics[model_name]['accuracy'].append(accuracy)
            
            # Keep only last 1000 metrics
            if len(self.model_metrics[model_name]['accuracy']) > 1000:
                self.model_metrics[model_name]['accuracy'] = self.model_metrics[model_name]['accuracy'][-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the model."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
                
            # Scale features
            X_scaled = self.scalers[model_name].transform(X)
            
            # Make predictions
            if self.model_type == 'sgd':
                return self.models[model_name].predict(X_scaled)
            else:
                return np.argmax(self.models[model_name].predict(X_scaled), axis=1)
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
            
    def save_model(self, model_name: str) -> bool:
        """Save model to disk."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
                
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save model
            if self.model_type == 'sgd':
                joblib.dump(self.models[model_name], os.path.join(self.model_dir, f"{model_name}.joblib"))
            else:
                self.models[model_name].save(os.path.join(self.model_dir, f"{model_name}.h5"))
                
            # Save scaler
            joblib.dump(self.scalers[model_name], os.path.join(self.model_dir, f"{model_name}_scaler.joblib"))
            
            # Save metrics
            joblib.dump(self.model_metrics[model_name], os.path.join(self.model_dir, f"{model_name}_metrics.joblib"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, model_name: str) -> bool:
        """Load model from disk."""
        try:
            # Check if model exists
            model_path = os.path.join(self.model_dir, f"{model_name}.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                if not os.path.exists(model_path):
                    return False
                    
            # Load model
            if model_path.endswith('.h5'):
                self.models[model_name] = load_model(model_path)
            else:
                self.models[model_name] = joblib.load(model_path)
                
            # Load scaler
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
                
            # Load metrics
            metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.joblib")
            if os.path.exists(metrics_path):
                self.model_metrics[model_name] = joblib.load(metrics_path)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
            
    def get_metrics(self, model_name: str) -> Dict:
        """Get model metrics."""
        try:
            if model_name not in self.model_metrics:
                return {}
                
            return self.model_metrics[model_name]
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {} 