import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ModelManager:
    """Manages AI models for efficient training and prediction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.models_dir = config.get('models_dir', 'models')
        self.model_configs = config.get('model_configs', {})
        self.max_models = config.get('max_models', 5)
        
        # Training settings
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}
        self._load_models()
        
    def _load_models(self):
        """Load existing models."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Load each model type
            for model_type in self.model_configs:
                model_path = os.path.join(self.models_dir, f"{model_type}.h5")
                if os.path.exists(model_path):
                    try:
                        self.models[model_type] = load_model(model_path)
                        self.logger.info(f"Loaded model: {model_type}")
                    except Exception as e:
                        self.logger.error(f"Error loading model {model_type}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            
    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> bool:
        """Train a model."""
        try:
            if model_type not in self.model_configs:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
                
            # Get model configuration
            config = self.model_configs[model_type]
            
            # Create model
            model = self._create_model(model_type, config, X.shape[1:])
            
            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    os.path.join(self.models_dir, f"{model_type}.h5"),
                    save_best_only=True,
                    monitor='val_loss'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model
            self.models[model_type] = model
            self.model_metrics[model_type] = {
                'last_trained': datetime.now(),
                'history': history.history
            }
            
            self.logger.info(f"Trained model: {model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model {model_type}: {str(e)}")
            return False
            
    def predict(self, model_type: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a model."""
        try:
            if model_type not in self.models:
                self.logger.error(f"Model not found: {model_type}")
                return np.array([])
                
            return self.models[model_type].predict(X)
            
        except Exception as e:
            self.logger.error(f"Error making predictions with model {model_type}: {str(e)}")
            return np.array([])
            
    def evaluate_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate a model."""
        try:
            if model_type not in self.models:
                self.logger.error(f"Model not found: {model_type}")
                return {}
                
            # Evaluate model
            metrics = self.models[model_type].evaluate(X, y, verbose=0)
            
            # Create metrics dictionary
            metric_names = self.models[model_type].metrics_names
            return dict(zip(metric_names, metrics))
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_type}: {str(e)}")
            return {}
            
    def _create_model(self, model_type: str, config: Dict, input_shape: tuple) -> tf.keras.Model:
        """Create a model based on configuration."""
        try:
            # Get model architecture
            architecture = config.get('architecture', [])
            
            # Create model
            model = tf.keras.Sequential()
            
            # Add input layer
            model.add(tf.keras.layers.Input(shape=input_shape))
            
            # Add layers
            for layer_config in architecture:
                layer_type = layer_config.get('type')
                layer_params = layer_config.get('params', {})
                
                if layer_type == 'dense':
                    model.add(tf.keras.layers.Dense(**layer_params))
                elif layer_type == 'conv1d':
                    model.add(tf.keras.layers.Conv1D(**layer_params))
                elif layer_type == 'lstm':
                    model.add(tf.keras.layers.LSTM(**layer_params))
                elif layer_type == 'dropout':
                    model.add(tf.keras.layers.Dropout(**layer_params))
                elif layer_type == 'batch_norm':
                    model.add(tf.keras.layers.BatchNormalization(**layer_params))
                    
            # Compile model
            model.compile(
                optimizer=config.get('optimizer', 'adam'),
                loss=config.get('loss', 'mse'),
                metrics=config.get('metrics', ['mae'])
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model {model_type}: {str(e)}")
            raise
            
    def get_model_info(self, model_type: str) -> Dict:
        """Get information about a model."""
        try:
            if model_type not in self.models:
                return {}
                
            model = self.models[model_type]
            metrics = self.model_metrics.get(model_type, {})
            
            return {
                'type': model_type,
                'architecture': model.get_config(),
                'last_trained': metrics.get('last_trained'),
                'history': metrics.get('history', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_type}: {str(e)}")
            return {}
            
    def cleanup_old_models(self):
        """Remove old models if exceeding max_models limit."""
        try:
            if len(self.models) <= self.max_models:
                return
                
            # Sort models by last trained time
            sorted_models = sorted(
                self.model_metrics.items(),
                key=lambda x: x[1].get('last_trained', datetime.min)
            )
            
            # Remove oldest models
            for model_type, _ in sorted_models[:-self.max_models]:
                model_path = os.path.join(self.models_dir, f"{model_type}.h5")
                if os.path.exists(model_path):
                    os.remove(model_path)
                del self.models[model_type]
                del self.model_metrics[model_type]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old models: {str(e)}") 