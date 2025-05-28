import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataManager:
    """Manages data for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data settings
        self.data_dir = config.get('data_dir', 'data')
        self.cache_dir = config.get('cache_dir', 'cache/data')
        self.max_cache_age = timedelta(hours=config.get('max_cache_age_hours', 24))
        
        # Feature settings
        self.feature_configs = config.get('feature_configs', {})
        self.scaler_configs = config.get('scaler_configs', {})
        
        # Initialize scalers
        self.scalers = {}
        self._load_scalers()
        
    def _load_scalers(self):
        """Load existing scalers."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load each scaler
            for scaler_type in self.scaler_configs:
                scaler_path = os.path.join(self.cache_dir, f"{scaler_type}_scaler.joblib")
                if os.path.exists(scaler_path):
                    try:
                        self.scalers[scaler_type] = joblib.load(scaler_path)
                        self.logger.info(f"Loaded scaler: {scaler_type}")
                    except Exception as e:
                        self.logger.error(f"Error loading scaler {scaler_type}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error loading scalers: {str(e)}")
            
    def load_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """Load data from file or cache."""
        try:
            # Check cache first
            cache_path = os.path.join(self.cache_dir, f"{data_type}.csv")
            if os.path.exists(cache_path):
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
                if cache_age < self.max_cache_age:
                    return pd.read_csv(cache_path, **kwargs)
                    
            # Load from data directory
            data_path = os.path.join(self.data_dir, f"{data_type}.csv")
            if not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return pd.DataFrame()
                
            # Load and cache data
            data = pd.read_csv(data_path, **kwargs)
            data.to_csv(cache_path, index=False)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data {data_type}: {str(e)}")
            return pd.DataFrame()
            
    def prepare_features(self, data: pd.DataFrame, feature_type: str) -> np.ndarray:
        """Prepare features for model input."""
        try:
            if feature_type not in self.feature_configs:
                self.logger.error(f"Unknown feature type: {feature_type}")
                return np.array([])
                
            # Get feature configuration
            config = self.feature_configs[feature_type]
            
            # Extract features
            features = []
            for feature in config.get('features', []):
                if feature in data.columns:
                    features.append(data[feature].values)
                    
            if not features:
                self.logger.error(f"No features found for {feature_type}")
                return np.array([])
                
            # Stack features
            X = np.column_stack(features)
            
            # Scale features if configured
            if config.get('scale', False):
                scaler_type = config.get('scaler_type', 'standard')
                if scaler_type not in self.scalers:
                    self.scalers[scaler_type] = self._create_scaler(scaler_type)
                X = self.scalers[scaler_type].transform(X)
                
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {feature_type}: {str(e)}")
            return np.array([])
            
    def _create_scaler(self, scaler_type: str):
        """Create a scaler based on configuration."""
        try:
            config = self.scaler_configs.get(scaler_type, {})
            
            if scaler_type == 'standard':
                return StandardScaler(**config)
            elif scaler_type == 'minmax':
                return MinMaxScaler(**config)
            else:
                self.logger.error(f"Unknown scaler type: {scaler_type}")
                return StandardScaler()
                
        except Exception as e:
            self.logger.error(f"Error creating scaler {scaler_type}: {str(e)}")
            return StandardScaler()
            
    def save_scalers(self):
        """Save current scalers to cache."""
        try:
            for scaler_type, scaler in self.scalers.items():
                scaler_path = os.path.join(self.cache_dir, f"{scaler_type}_scaler.joblib")
                joblib.dump(scaler, scaler_path)
                
        except Exception as e:
            self.logger.error(f"Error saving scalers: {str(e)}")
            
    def get_feature_info(self, feature_type: str) -> Dict:
        """Get information about features."""
        try:
            if feature_type not in self.feature_configs:
                return {}
                
            config = self.feature_configs[feature_type]
            
            return {
                'type': feature_type,
                'features': config.get('features', []),
                'scale': config.get('scale', False),
                'scaler_type': config.get('scaler_type', 'standard')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feature info for {feature_type}: {str(e)}")
            return {}
            
    def cleanup_cache(self):
        """Remove old cache files."""
        try:
            if not os.path.exists(self.cache_dir):
                return
                
            # Get current time
            now = datetime.now()
            
            # Check each cache file
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                file_age = now - datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Remove old files
                if file_age > self.max_cache_age:
                    os.remove(filepath)
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {str(e)}") 