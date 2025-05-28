import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import os

from ..core.base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """Neural Network model for trading predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('neural_network', {})
        self.input_shape = None
        self.predict_fn = None
        
    def initialize(self):
        """Initialize the neural network model."""
        try:
            self.model = self._create_model()
            self.scaler = StandardScaler()
            self.logger.info("Neural Network model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Neural Network model: {str(e)}")
            raise
            
    def _create_model(self) -> Sequential:
        """Create the neural network architecture."""
        try:
            model = Sequential([
                layers.Dense(128, activation='relu', input_shape=self.input_shape),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating neural network model: {str(e)}")
            raise
            
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for the neural network."""
        try:
            # Calculate all feature sets
            data = self._calculate_technical_indicators(data)
            data = self._calculate_microstructure_features(data)
            data = self._calculate_sentiment_features(data)
            data = self._calculate_volatility_features(data)
            
            # Remove any NaN values
            data = data.fillna(0)
            
            # Scale the features
            if self.scaler is None:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(data)
            else:
                features = self.scaler.transform(data)
                
            # Set input shape if not set
            if self.input_shape is None:
                self.input_shape = (features.shape[1],)
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the neural network model."""
        try:
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    'models/neural_network_best.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train the model
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Create prediction function
            self.predict_fn = self._create_keras_predict_fn(self.model)
            
            # Update performance metrics
            self.performance_metrics['training_history'] = history.history
            
            self.logger.info("Neural Network model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training neural network model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the neural network."""
        try:
            if self.predict_fn is None:
                self.predict_fn = self._create_keras_predict_fn(self.model)
                
            predictions = self.predict_fn(X)
            return predictions.numpy()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the neural network model."""
        try:
            # Get predictions
            y_pred = self.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': np.mean(y_pred_binary == y_test),
                'precision': np.sum((y_pred_binary == 1) & (y_test == 1)) / np.sum(y_pred_binary == 1),
                'recall': np.sum((y_pred_binary == 1) & (y_test == 1)) / np.sum(y_test == 1)
            }
            
            # Update performance metrics
            self.performance_metrics.update(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save(self, path: str):
        """Save the neural network model and scaler."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            self.model.save(f"{path}_model.h5")
            
            # Save scaler
            joblib.dump(self.scaler, f"{path}_scaler.joblib")
            
            # Save performance metrics
            np.save(f"{path}_metrics.npy", self.performance_metrics)
            
            self.logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load(self, path: str):
        """Load the neural network model and scaler."""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{path}_model.h5")
            
            # Load scaler
            self.scaler = joblib.load(f"{path}_scaler.joblib")
            
            # Load performance metrics
            self.performance_metrics = np.load(f"{path}_metrics.npy", allow_pickle=True).item()
            
            # Create prediction function
            self.predict_fn = self._create_keras_predict_fn(self.model)
            
            self.logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise 