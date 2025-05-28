import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os

class ModelPipeline:
    """Geavanceerde model pipeline voor AI modellen."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.model_dir = config.get('model_dir', 'models')
        self.model_configs = config.get('model_configs', {})
        
        # Training settings
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}
        
    def prepare_data(self, data: pd.DataFrame, target: str, test_size: float = 0.2) -> tuple:
        """Prepare data for training."""
        try:
            # Split features and target
            X = data.drop(columns=[target])
            y = data[target]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None, None, None
            
    def train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train a model."""
        try:
            if model_type not in self.model_configs:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
                
            # Get model configuration
            config = self.model_configs[model_type]
            
            # Create and train model
            if config.get('type') == 'neural_network':
                model = self._create_neural_network(config, X.shape[1:])
                history = self._train_neural_network(model, X, y)
            else:
                model = self._create_sklearn_model(config)
                model.fit(X, y)
                history = None
                
            # Save model
            self.models[model_type] = model
            self.model_metrics[model_type] = {
                'history': history.history if history else None,
                'last_trained': pd.Timestamp.now()
            }
            
            # Save to disk
            self._save_model(model_type, model)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model {model_type}: {str(e)}")
            return False
            
    def _create_neural_network(self, config: Dict, input_shape: tuple) -> tf.keras.Model:
        """Create a neural network model."""
        try:
            # Create model
            model = Sequential()
            
            # Add input layer
            model.add(Dense(
                units=config.get('hidden_units', 64),
                activation=config.get('activation', 'relu'),
                input_shape=input_shape
            ))
            
            # Add hidden layers
            for _ in range(config.get('hidden_layers', 2)):
                model.add(Dense(
                    units=config.get('hidden_units', 64),
                    activation=config.get('activation', 'relu')
                ))
                model.add(BatchNormalization())
                model.add(Dropout(config.get('dropout_rate', 0.2)))
                
            # Add output layer
            model.add(Dense(
                units=config.get('output_units', 1),
                activation=config.get('output_activation', 'sigmoid')
            ))
            
            # Compile model
            model.compile(
                optimizer=config.get('optimizer', 'adam'),
                loss=config.get('loss', 'binary_crossentropy'),
                metrics=config.get('metrics', ['accuracy'])
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating neural network: {str(e)}")
            raise
            
    def _train_neural_network(self, model: tf.keras.Model, X: pd.DataFrame, y: pd.Series) -> tf.keras.callbacks.History:
        """Train a neural network model."""
        try:
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'best_model.h5'),
                    save_best_only=True,
                    monitor='val_loss'
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
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {str(e)}")
            raise
            
    def _create_sklearn_model(self, config: Dict):
        """Create a scikit-learn model."""
        try:
            model_type = config.get('type')
            
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**config.get('params', {}))
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**config.get('params', {}))
            elif model_type == 'svm':
                from sklearn.svm import SVC
                return SVC(**config.get('params', {}))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating sklearn model: {str(e)}")
            raise
            
    def _save_model(self, model_type: str, model: Any):
        """Save model to disk."""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save model
            if isinstance(model, tf.keras.Model):
                model.save(os.path.join(self.model_dir, f"{model_type}.h5"))
            else:
                joblib.dump(model, os.path.join(self.model_dir, f"{model_type}.joblib"))
                
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load_model(self, model_type: str) -> bool:
        """Load model from disk."""
        try:
            # Check if model exists
            model_path = os.path.join(self.model_dir, f"{model_type}.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, f"{model_type}.joblib")
                if not os.path.exists(model_path):
                    return False
                    
            # Load model
            if model_path.endswith('.h5'):
                self.models[model_type] = load_model(model_path)
            else:
                self.models[model_type] = joblib.load(model_path)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
            
    def predict(self, model_type: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with a model."""
        try:
            if model_type not in self.models:
                if not self.load_model(model_type):
                    raise ValueError(f"Model not found: {model_type}")
                    
            return self.models[model_type].predict(X)
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
            
    def evaluate_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance."""
        try:
            if model_type not in self.models:
                if not self.load_model(model_type):
                    raise ValueError(f"Model not found: {model_type}")
                    
            # Make predictions
            y_pred = self.predict(model_type, X)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
            
    def optimize_hyperparameters(self, model_type: str, X: pd.DataFrame, y: pd.Series, param_grid: Dict) -> Dict:
        """Optimize model hyperparameters."""
        try:
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
                
            # Create model
            model = self._create_sklearn_model(self.model_configs[model_type])
            
            # Setup grid search
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Perform grid search
            grid_search.fit(X, y)
            
            # Update model configuration
            self.model_configs[model_type]['params'] = grid_search.best_params_
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters: {str(e)}")
            return {} 