import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,
    Flatten, Dropout, BatchNormalization, Input, Concatenate,
    Bidirectional, Attention, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
import os

class DeepLearning:
    """Deep learning extensies met state-of-the-art modellen."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.model_dir = config.get('model_dir', 'models')
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}
        
    def create_cnn_model(self, model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> bool:
        """Create a CNN model."""
        try:
            # Create model
            model = Sequential([
                # First convolutional block
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Third convolutional block
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Dense layers
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save model
            self.models[model_name] = model
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating CNN model: {str(e)}")
            return False
            
    def create_rnn_model(self, model_name: str, input_shape: Tuple[int, int], num_classes: int) -> bool:
        """Create an RNN model."""
        try:
            # Create model
            model = Sequential([
                # Bidirectional LSTM layers
                Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.3),
                
                Bidirectional(LSTM(64, return_sequences=True)),
                BatchNormalization(),
                Dropout(0.3),
                
                Bidirectional(LSTM(32)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Dense layers
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save model
            self.models[model_name] = model
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating RNN model: {str(e)}")
            return False
            
    def create_transformer_model(self, model_name: str, input_shape: Tuple[int, int], num_classes: int) -> bool:
        """Create a Transformer model."""
        try:
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Multi-head attention layers
            attention_output = MultiHeadAttention(
                num_heads=8,
                key_dim=64
            )(inputs, inputs)
            attention_output = LayerNormalization()(attention_output + inputs)
            
            # Feed-forward network
            ffn_output = Dense(512, activation='relu')(attention_output)
            ffn_output = Dense(input_shape[1])(ffn_output)
            ffn_output = LayerNormalization()(ffn_output + attention_output)
            
            # Global average pooling and dense layers
            x = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save model
            self.models[model_name] = model
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating Transformer model: {str(e)}")
            return False
            
    def create_transfer_learning_model(self, model_name: str, base_model: str, input_shape: Tuple[int, int, int], num_classes: int) -> bool:
        """Create a transfer learning model."""
        try:
            # Load base model
            if base_model == 'resnet50':
                base = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=input_shape
                )
            elif base_model == 'vgg16':
                base = VGG16(
                    weights='imagenet',
                    include_top=False,
                    input_shape=input_shape
                )
            elif base_model == 'efficientnet':
                base = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=input_shape
                )
            else:
                raise ValueError(f"Unknown base model: {base_model}")
                
            # Freeze base model layers
            base.trainable = False
            
            # Create model
            model = Sequential([
                base,
                GlobalAveragePooling2D(),
                Dense(1024, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save model
            self.models[model_name] = model
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating transfer learning model: {str(e)}")
            return False
            
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> bool:
        """Train a model."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
                
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    os.path.join(self.model_dir, f"{model_name}_best.h5"),
                    save_best_only=True,
                    monitor='val_loss'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = self.models[model_name].fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save metrics
            self.model_metrics[model_name] = {
                'history': history.history,
                'last_trained': pd.Timestamp.now()
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False
            
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
                
            return self.models[model_name].predict(X)
            
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
            self.models[model_name].save(os.path.join(self.model_dir, f"{model_name}.h5"))
            
            # Save metrics
            if model_name in self.model_metrics:
                joblib.dump(
                    self.model_metrics[model_name],
                    os.path.join(self.model_dir, f"{model_name}_metrics.joblib")
                )
                
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
                return False
                
            # Load model
            self.models[model_name] = load_model(model_path)
            
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