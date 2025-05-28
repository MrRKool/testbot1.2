import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import pandas_ta as ta
from scipy import stats

class FeatureEngineering:
    """Geavanceerde feature engineering voor AI modellen."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature settings
        self.feature_configs = config.get('feature_configs', {})
        self.scaler_configs = config.get('scaler_configs', {})
        
        # Initialize components
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_transformers = {}
        
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features."""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Remove outliers
            data = self._remove_outliers(data)
            
            # Normalize data
            data = self._normalize_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing features: {str(e)}")
            return data
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data."""
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Fill missing values
            for col in numeric_cols:
                if data[col].isnull().any():
                    # Use median for numeric columns
                    data[col] = data[col].fillna(data[col].median())
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return data
            
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data."""
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Remove outliers using IQR method
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                data[col] = data[col].clip(lower_bound, upper_bound)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing outliers: {str(e)}")
            return data
            
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using appropriate scaler."""
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Apply scaler
            for col in numeric_cols:
                if col not in self.scalers:
                    self.scalers[col] = self._create_scaler(col)
                data[col] = self.scalers[col].fit_transform(data[col].values.reshape(-1, 1))
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {str(e)}")
            return data
            
    def _create_scaler(self, column: str):
        """Create appropriate scaler for column."""
        try:
            config = self.scaler_configs.get(column, {})
            scaler_type = config.get('type', 'standard')
            
            if scaler_type == 'standard':
                return StandardScaler(**config)
            elif scaler_type == 'minmax':
                return MinMaxScaler(**config)
            elif scaler_type == 'robust':
                return RobustScaler(**config)
            else:
                return StandardScaler()
                
        except Exception as e:
            self.logger.error(f"Error creating scaler: {str(e)}")
            return StandardScaler()
            
    def select_features(self, data: pd.DataFrame, target: pd.Series, method: str = 'f_classif', k: int = 10) -> pd.DataFrame:
        """Select most important features."""
        try:
            if method not in self.feature_selectors:
                if method == 'f_classif':
                    self.feature_selectors[method] = SelectKBest(f_classif, k=k)
                elif method == 'mutual_info':
                    self.feature_selectors[method] = SelectKBest(mutual_info_classif, k=k)
                    
            # Select features
            selected_features = self.feature_selectors[method].fit_transform(data, target)
            
            # Get selected feature names
            selected_indices = self.feature_selectors[method].get_support(indices=True)
            selected_columns = data.columns[selected_indices]
            
            return data[selected_columns]
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return data
            
    def transform_features(self, data: pd.DataFrame, method: str = 'pca', n_components: int = 2) -> pd.DataFrame:
        """Transform features using dimensionality reduction."""
        try:
            if method not in self.feature_transformers:
                if method == 'pca':
                    self.feature_transformers[method] = PCA(n_components=n_components)
                    
            # Transform features
            transformed_features = self.feature_transformers[method].fit_transform(data)
            
            # Create DataFrame with transformed features
            transformed_df = pd.DataFrame(
                transformed_features,
                columns=[f'{method}_{i}' for i in range(n_components)]
            )
            
            return transformed_df
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {str(e)}")
            return data
            
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators."""
        try:
            # Get OHLCV data
            open_price = data['open'].values
            high_price = data['high'].values
            low_price = data['low'].values
            close_price = data['close'].values
            volume = data['volume'].values
            
            # Create indicators
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = data.ta.sma(length=20)
            indicators['ema_20'] = data.ta.ema(length=20)
            
            # Oscillators
            indicators['rsi'] = data.ta.rsi(length=14)
            macd = data.ta.macd()
            indicators['macd'] = macd['MACD_12_26_9']
            indicators['macd_signal'] = macd['MACDs_12_26_9']
            indicators['macd_hist'] = macd['MACDh_12_26_9']
            
            # Volatility
            indicators['atr'] = data.ta.atr(length=14)
            indicators['natr'] = data.ta.natr(length=14)
            
            # Volume
            indicators['obv'] = data.ta.obv()
            
            # Create DataFrame
            indicators_df = pd.DataFrame(indicators, index=data.index)
            
            return indicators_df
            
        except Exception as e:
            self.logger.error(f"Error creating technical indicators: {str(e)}")
            return pd.DataFrame()
            
    def create_statistical_features(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Create statistical features."""
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Create statistical features
            stats_features = {}
            
            for col in numeric_cols:
                # Rolling statistics
                stats_features[f'{col}_rolling_mean'] = data[col].rolling(window=window).mean()
                stats_features[f'{col}_rolling_std'] = data[col].rolling(window=window).std()
                stats_features[f'{col}_rolling_skew'] = data[col].rolling(window=window).skew()
                stats_features[f'{col}_rolling_kurt'] = data[col].rolling(window=window).kurt()
                
                # Z-score
                stats_features[f'{col}_zscore'] = stats.zscore(data[col])
                
            # Create DataFrame
            stats_df = pd.DataFrame(stats_features, index=data.index)
            
            return stats_df
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            return pd.DataFrame()
            
    def get_feature_importance(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Get feature importance scores."""
        try:
            # Create feature selector
            selector = SelectKBest(f_classif, k='all')
            
            # Fit selector
            selector.fit(data, target)
            
            # Get scores
            scores = selector.scores_
            
            # Create importance dictionary
            importance = dict(zip(data.columns, scores))
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {} 