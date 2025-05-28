import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from scipy import stats
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import layers, models
import optuna
from sklearn.preprocessing import StandardScaler
from ..utils.risk_utils import calculate_var, calculate_expected_shortfall

class MathematicalAI:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Mathematical AI module"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.scaler = StandardScaler()
        self.optimization_history = []
        self.strategy_weights = {}
        self.risk_parameters = {}
        
        # Initialize mathematical models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize mathematical models for different aspects of trading"""
        # Statistical models
        self.statistical_models = {
            'regression': self._create_regression_model(),
            'time_series': self._create_time_series_model(),
            'volatility': self._create_volatility_model()
        }
        
        # Optimization models
        self.optimization_models = {
            'portfolio': self._create_portfolio_optimizer(),
            'risk': self._create_risk_optimizer()
        }
        
        # Strategy models
        self.strategy_models = {
            'trend': self._create_trend_model(),
            'mean_reversion': self._create_mean_reversion_model(),
            'breakout': self._create_breakout_model()
        }
        
    def _create_regression_model(self):
        """Create a regression model for price prediction"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(50,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _create_time_series_model(self):
        """Create a time series model for trend analysis"""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(50, 1)),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _create_volatility_model(self):
        """Create a volatility model for risk assessment"""
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(50,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    def _create_portfolio_optimizer(self):
        """Create a portfolio optimization model"""
        def objective(weights, returns, cov_matrix):
            portfolio_return = np.sum(returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_risk
            return -sharpe_ratio
            
        return objective
        
    def _create_risk_optimizer(self):
        """Create a risk optimization model"""
        def objective(params, data):
            var = calculate_var(data, params['confidence_level'])
            es = calculate_expected_shortfall(data, params['confidence_level'])
            return var + es
            
        return objective
        
    def _create_trend_model(self):
        """Create a trend following model"""
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(50,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    def _create_mean_reversion_model(self):
        """Create a mean reversion model"""
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(50,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    def _create_breakout_model(self):
        """Create a breakout detection model"""
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(50,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
        
    def optimize_strategy_weights(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize weights for different trading strategies"""
        try:
            # Calculate returns for each strategy
            strategy_returns = {}
            for name, model in self.strategy_models.items():
                predictions = model.predict(historical_data)
                strategy_returns[name] = self._calculate_strategy_returns(predictions, historical_data)
                
            # Calculate covariance matrix
            returns_df = pd.DataFrame(strategy_returns)
            cov_matrix = returns_df.cov()
            
            # Optimize weights
            n_strategies = len(self.strategy_models)
            initial_weights = np.array([1/n_strategies] * n_strategies)
            bounds = tuple((0, 1) for _ in range(n_strategies))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            result = minimize(
                self.optimization_models['portfolio'],
                initial_weights,
                args=(returns_df.mean(), cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Store optimized weights
            self.strategy_weights = dict(zip(self.strategy_models.keys(), result.x))
            
            return self.strategy_weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy weights: {str(e)}")
            return {}
            
    def optimize_risk_parameters(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize risk management parameters"""
        try:
            # Define parameter bounds
            param_bounds = {
                'position_size': (0.01, 0.1),
                'stop_loss': (0.01, 0.05),
                'take_profit': (0.02, 0.1),
                'max_drawdown': (0.05, 0.2)
            }
            
            # Create optimization study
            study = optuna.create_study(direction='minimize')
            
            def objective(trial):
                params = {
                    'position_size': trial.suggest_float('position_size', *param_bounds['position_size']),
                    'stop_loss': trial.suggest_float('stop_loss', *param_bounds['stop_loss']),
                    'take_profit': trial.suggest_float('take_profit', *param_bounds['take_profit']),
                    'max_drawdown': trial.suggest_float('max_drawdown', *param_bounds['max_drawdown'])
                }
                
                return self.optimization_models['risk'](params, market_data)
                
            # Run optimization
            study.optimize(objective, n_trials=100)
            
            # Store optimized parameters
            self.risk_parameters = study.best_params
            
            return self.risk_parameters
            
        except Exception as e:
            self.logger.error(f"Error optimizing risk parameters: {str(e)}")
            return {}
            
    def calculate_mathematical_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mathematical trading signals"""
        try:
            signals = {}
            
            # Calculate statistical signals
            signals['regression'] = self._calculate_regression_signals(market_data)
            signals['time_series'] = self._calculate_time_series_signals(market_data)
            signals['volatility'] = self._calculate_volatility_signals(market_data)
            
            # Calculate strategy signals
            signals['trend'] = self._calculate_trend_signals(market_data)
            signals['mean_reversion'] = self._calculate_mean_reversion_signals(market_data)
            signals['breakout'] = self._calculate_breakout_signals(market_data)
            
            # Combine signals using optimized weights
            combined_signal = self._combine_signals(signals)
            
            return {
                'signals': signals,
                'combined_signal': combined_signal,
                'confidence': self._calculate_signal_confidence(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating mathematical signals: {str(e)}")
            return {}
            
    def _calculate_regression_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using regression model"""
        try:
            features = self._prepare_features(data)
            predictions = self.statistical_models['regression'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating regression signals: {str(e)}")
            return np.zeros(len(data))
            
    def _calculate_time_series_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using time series model"""
        try:
            features = self._prepare_features(data)
            predictions = self.statistical_models['time_series'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating time series signals: {str(e)}")
            return np.zeros(len(data))
            
    def _calculate_volatility_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using volatility model"""
        try:
            features = self._prepare_features(data)
            predictions = self.statistical_models['volatility'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating volatility signals: {str(e)}")
            return np.zeros(len(data))
            
    def _calculate_trend_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using trend model"""
        try:
            features = self._prepare_features(data)
            predictions = self.strategy_models['trend'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating trend signals: {str(e)}")
            return np.zeros(len(data))
            
    def _calculate_mean_reversion_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using mean reversion model"""
        try:
            features = self._prepare_features(data)
            predictions = self.strategy_models['mean_reversion'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion signals: {str(e)}")
            return np.zeros(len(data))
            
    def _calculate_breakout_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate signals using breakout model"""
        try:
            features = self._prepare_features(data)
            predictions = self.strategy_models['breakout'].predict(features)
            return predictions.flatten()
        except Exception as e:
            self.logger.error(f"Error calculating breakout signals: {str(e)}")
            return np.zeros(len(data))
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # Calculate technical indicators
            features = pd.DataFrame()
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['momentum'] = data['close'].pct_change(periods=10)
            features['rsi'] = self._calculate_rsi(data['close'])
            features['macd'] = self._calculate_macd(data['close'])
            
            # Handle missing values
            features = features.fillna(0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return np.zeros((len(data), 5))
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate Moving Average Convergence Divergence"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _combine_signals(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using optimized weights"""
        try:
            combined = np.zeros(len(next(iter(signals.values()))))
            for strategy, signal in signals.items():
                if strategy in self.strategy_weights:
                    combined += signal * self.strategy_weights[strategy]
            return combined
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            return np.zeros(len(next(iter(signals.values()))))
            
    def _calculate_signal_confidence(self, signals: Dict[str, np.ndarray]) -> float:
        """Calculate confidence in combined signal"""
        try:
            # Calculate correlation between different signals
            signal_df = pd.DataFrame(signals)
            correlation_matrix = signal_df.corr()
            
            # Calculate average correlation
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()
            
            # Calculate signal strength
            signal_strength = np.mean([np.abs(signal).mean() for signal in signals.values()])
            
            # Combine metrics
            confidence = (1 + avg_correlation) * signal_strength / 2
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {str(e)}")
            return 0.0
            
    def _calculate_strategy_returns(self, predictions: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """Calculate returns for a strategy based on predictions"""
        try:
            returns = data['close'].pct_change().values[1:]
            strategy_returns = np.where(predictions[:-1] > 0.5, returns, -returns)
            return strategy_returns
        except Exception as e:
            self.logger.error(f"Error calculating strategy returns: {str(e)}")
            return np.zeros(len(data) - 1)
            
    def save_state(self):
        """Save the state of all models"""
        try:
            # Save statistical models
            for name, model in self.statistical_models.items():
                model.save(f"models/statistical_{name}.h5")
                
            # Save strategy models
            for name, model in self.strategy_models.items():
                model.save(f"models/strategy_{name}.h5")
                
            # Save scaler
            import joblib
            joblib.dump(self.scaler, "models/scaler.pkl")
            
            # Save optimization history
            pd.DataFrame(self.optimization_history).to_csv("models/optimization_history.csv")
            
            # Save strategy weights and risk parameters
            import json
            with open("models/strategy_weights.json", "w") as f:
                json.dump(self.strategy_weights, f)
            with open("models/risk_parameters.json", "w") as f:
                json.dump(self.risk_parameters, f)
                
            self.logger.info("Saved mathematical AI state")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            
    def load_state(self):
        """Load the state of all models"""
        try:
            # Load statistical models
            for name in self.statistical_models.keys():
                path = f"models/statistical_{name}.h5"
                if os.path.exists(path):
                    self.statistical_models[name] = models.load_model(path)
                    
            # Load strategy models
            for name in self.strategy_models.keys():
                path = f"models/strategy_{name}.h5"
                if os.path.exists(path):
                    self.strategy_models[name] = models.load_model(path)
                    
            # Load scaler
            import joblib
            if os.path.exists("models/scaler.pkl"):
                self.scaler = joblib.load("models/scaler.pkl")
                
            # Load optimization history
            if os.path.exists("models/optimization_history.csv"):
                self.optimization_history = pd.read_csv("models/optimization_history.csv").to_dict('records')
                
            # Load strategy weights and risk parameters
            import json
            if os.path.exists("models/strategy_weights.json"):
                with open("models/strategy_weights.json", "r") as f:
                    self.strategy_weights = json.load(f)
            if os.path.exists("models/risk_parameters.json"):
                with open("models/risk_parameters.json", "r") as f:
                    self.risk_parameters = json.load(f)
                    
            self.logger.info("Loaded mathematical AI state")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}") 