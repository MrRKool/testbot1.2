from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import pandas_ta as ta
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def calculate_features(data: pd.DataFrame) -> np.ndarray:
    """Calculate features for price prediction."""
    try:
        features = []
        
        # Price features
        features.append(data['close'].values)
        features.append(data['volume'].values)
        
        # Technical indicators
        features.append(data.ta.rsi(length=14))
        macd = data.ta.macd()
        features.append(macd['MACD_12_26_9'])
        bbands = data.ta.bbands()
        features.append(bbands['BBU_20_2.0'])
        
        return np.column_stack(features)
        
    except Exception as e:
        raise ValueError(f"Error calculating features: {e}")

def calculate_regime_features(data: pd.DataFrame) -> np.ndarray:
    """Calculate features for regime detection."""
    try:
        features = []
        
        # Volatility
        features.append(data['close'].pct_change().std())
        
        # Trend
        features.append(ta.adx(data['high'], data['low'], data['close'])[-1])
        
        # Volume
        features.append(data['volume'].mean() / data['volume'].std())
        
        # Momentum
        features.append(data.ta.rsi(length=14)[-1])
        
        return np.array(features)
        
    except Exception as e:
        raise ValueError(f"Error calculating regime features: {e}")

def calculate_shrinkage_covariance(returns: pd.DataFrame, shrinkage_factor: float) -> np.ndarray:
    """Calculate covariance matrix with shrinkage."""
    try:
        # Sample covariance
        sample_cov = returns.cov()
        
        # Shrinkage target (constant correlation)
        n = returns.shape[1]
        mean_corr = (sample_cov.sum() - n) / (n * (n - 1))
        target = np.ones((n, n)) * mean_corr
        np.fill_diagonal(target, 1)
        
        # Shrinkage
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
        
        return shrunk_cov
        
    except Exception as e:
        raise ValueError(f"Error calculating shrinkage covariance: {e}")

def calculate_risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Calculate risk parity weights."""
    try:
        n = cov_matrix.shape[0]
        weights = np.ones(n) / n
        
        def risk_contribution(w):
            portfolio_risk = np.sqrt(w.T @ cov_matrix @ w)
            risk_contrib = (cov_matrix @ w) * w / portfolio_risk
            return np.sum((risk_contrib - portfolio_risk/n)**2)
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        result = minimize(
            risk_contribution,
            weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
        
    except Exception as e:
        raise ValueError(f"Error calculating risk parity weights: {e}")

def calculate_mpt_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Calculate Modern Portfolio Theory weights."""
    try:
        n = cov_matrix.shape[0]
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        
        weights = (inv_cov_matrix @ ones) / (ones @ inv_cov_matrix @ ones)
        
        return weights
        
    except Exception as e:
        raise ValueError(f"Error calculating MPT weights: {e}")

def apply_portfolio_constraints(weights: np.ndarray) -> np.ndarray:
    """Apply portfolio constraints."""
    try:
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Apply minimum and maximum weights
        weights = np.clip(weights, 0.01, 0.5)
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights
        
    except Exception as e:
        raise ValueError(f"Error applying portfolio constraints: {e}")

def check_missing_values(data: pd.DataFrame, max_missing_ratio: float) -> bool:
    """Check for missing values."""
    try:
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        return missing_ratio <= max_missing_ratio
        
    except Exception as e:
        raise ValueError(f"Error checking missing values: {e}")

def check_outliers(data: pd.DataFrame) -> bool:
    """Check for outliers."""
    try:
        for column in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(data[column]))
            if (z_scores > 3).any():
                return False
        return True
        
    except Exception as e:
        raise ValueError(f"Error checking outliers: {e}")

def check_data_consistency(data: pd.DataFrame) -> bool:
    """Check data consistency."""
    try:
        # Check for price gaps
        price_gaps = data['close'].pct_change().abs() > 0.1
        if price_gaps.any():
            return False
            
        # Check for volume consistency
        volume_consistency = data['volume'] > 0
        if not volume_consistency.all():
            return False
            
        # Check for time consistency
        time_diff = data.index.to_series().diff()
        if not (time_diff == pd.Timedelta('1min')).all():
            return False
            
        return True
        
    except Exception as e:
        raise ValueError(f"Error checking data consistency: {e}")

def generate_warnings(results: Dict[str, bool]) -> List[str]:
    """Generate warnings from validation results."""
    try:
        warnings = []
        
        if not results['missing']:
            warnings.append("High ratio of missing values")
            
        if not results['outliers']:
            warnings.append("Outliers detected in data")
            
        if not results['consistency']:
            warnings.append("Data consistency issues detected")
            
        return warnings
        
    except Exception as e:
        raise ValueError(f"Error generating warnings: {e}")

def calculate_metrics(trades: List[Dict[str, Any]], equity_curve: List[float]) -> Dict[str, float]:
    """Calculate performance metrics."""
    try:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            "total_trades": len(trades),
            "winning_trades": sum(1 for t in trades if t["pnl"] > 0),
            "losing_trades": sum(1 for t in trades if t["pnl"] <= 0),
            "total_pnl": sum(t["pnl"] for t in trades),
            "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades) if trades else 0,
            "average_win": np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if any(t["pnl"] > 0 for t in trades) else 0,
            "average_loss": np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if any(t["pnl"] <= 0 for t in trades) else 0,
            "max_drawdown": calculate_max_drawdown(equity_curve),
            "sharpe_ratio": calculate_sharpe_ratio(returns)
        }
        
        return metrics
        
    except Exception as e:
        raise ValueError(f"Error calculating metrics: {e}")

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown."""
    try:
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    except Exception as e:
        raise ValueError(f"Error calculating max drawdown: {e}")

def calculate_sharpe_ratio(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio."""
    try:
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
    except Exception as e:
        raise ValueError(f"Error calculating Sharpe ratio: {e}")

def prepare_training_data(data: pd.DataFrame, training_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for training."""
    try:
        # Calculate features
        features = calculate_features(data)
        
        # Create sequences
        X = np.lib.stride_tricks.sliding_window_view(
            features,
            (training_window, features.shape[1])
        )
        y = features[training_window:, 0]
        
        # Normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return X, y
        
    except Exception as e:
        raise ValueError(f"Error preparing training data: {e}")

def save_models(models: Dict[str, Any], model_dir: str):
    """Save trained models."""
    try:
        # Save price prediction model
        models["price_prediction"].save(
            os.path.join(model_dir, "price_prediction.h5")
        )
        
        # Save regime classification model
        joblib.dump(
            models["regime_classification"],
            os.path.join(model_dir, "regime_classification.pkl")
        )
        
    except Exception as e:
        raise ValueError(f"Error saving models: {e}")

def load_models(model_dir: str) -> Dict[str, Any]:
    """Load trained models."""
    try:
        models = {}
        
        # Load price prediction model
        model_path = os.path.join(model_dir, "price_prediction.h5")
        if os.path.exists(model_path):
            models["price_prediction"] = tf.keras.models.load_model(model_path)
            
        # Load regime classification model
        model_path = os.path.join(model_dir, "regime_classification.pkl")
        if os.path.exists(model_path):
            models["regime_classification"] = joblib.load(model_path)
            
        return models
        
    except Exception as e:
        raise ValueError(f"Error loading models: {e}") 