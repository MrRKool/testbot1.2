import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from utils.advanced_features import AdvancedFeatures, AdvancedConfig
from utils.feature_helpers import (
    calculate_features,
    calculate_regime_features,
    calculate_shrinkage_covariance,
    calculate_risk_parity_weights,
    calculate_mpt_weights,
    apply_portfolio_constraints,
    check_missing_values,
    check_outliers,
    check_data_consistency,
    generate_warnings,
    calculate_metrics,
    prepare_training_data,
    save_models,
    load_models
)

@pytest.fixture
def config():
    """Create test configuration."""
    return AdvancedConfig(
        ml_model_dir='test_models',
        backtest_dir='test_backtests',
        require_2fa=False,
        ip_whitelist=['127.0.0.1'],
        alert_thresholds={
            'max_drawdown': 0.1,
            'min_sharpe': 1.0,
            'max_position_size': 0.2
        },
        cache_enabled=False,
        use_redis=False,
        enable_profiling=False
    )

@pytest.fixture
def features(config):
    """Create test features instance."""
    return AdvancedFeatures(config)

@pytest.fixture
def test_data():
    """Create test data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    
    # Add some trends
    data['close'] = data['close'] + np.linspace(0, 10, len(dates))
    data['high'] = data['high'] + np.linspace(0, 10, len(dates))
    data['low'] = data['low'] + np.linspace(0, 10, len(dates))
    
    return data

@pytest.fixture
def test_returns():
    """Create test returns data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1D')
    returns = pd.DataFrame({
        'BTC': np.random.normal(0.001, 0.02, len(dates)),
        'ETH': np.random.normal(0.001, 0.02, len(dates)),
        'BNB': np.random.normal(0.001, 0.02, len(dates))
    }, index=dates)
    
    return returns

@pytest.fixture
def test_trades():
    """Create test trades data."""
    return [
        {
            'entry_time': datetime(2023, 1, 1, 10, 0),
            'exit_time': datetime(2023, 1, 1, 11, 0),
            'entry_price': 100,
            'exit_price': 110,
            'size': 1,
            'pnl': 10
        },
        {
            'entry_time': datetime(2023, 1, 1, 12, 0),
            'exit_time': datetime(2023, 1, 1, 13, 0),
            'entry_price': 110,
            'exit_price': 105,
            'size': 1,
            'pnl': -5
        }
    ]

@pytest.fixture
def test_equity_curve():
    """Create test equity curve."""
    return [100, 110, 105, 115, 120]

def test_init(features):
    """Test initialization."""
    assert features is not None
    assert features.config is not None
    assert features.models is not None

def test_train_models(features, test_data):
    """Test model training."""
    features.train_models(test_data)
    assert 'price_prediction' in features.models
    assert 'regime_classification' in features.models

def test_predict_price(features, test_data):
    """Test price prediction."""
    features.train_models(test_data)
    prediction = features.predict_price(test_data)
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)

def test_detect_regime(features, test_data):
    """Test regime detection."""
    features.train_models(test_data)
    regime = features.detect_regime(test_data)
    assert regime in ['bullish', 'bearish', 'sideways']

def test_optimize_portfolio(features, test_returns):
    """Test portfolio optimization."""
    weights = features.optimize_portfolio(test_returns)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(test_returns.columns)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)

def test_validate_data(features, test_data):
    """Test data validation."""
    validation = features.validate_data(test_data)
    assert isinstance(validation, dict)
    assert 'valid' in validation
    assert 'warnings' in validation

def test_calculate_performance_metrics(features, test_trades, test_equity_curve):
    """Test performance metrics calculation."""
    metrics = features.calculate_performance_metrics(test_trades, test_equity_curve)
    assert isinstance(metrics, dict)
    assert 'total_trades' in metrics
    assert 'winning_trades' in metrics
    assert 'losing_trades' in metrics
    assert 'total_pnl' in metrics
    assert 'win_rate' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics

def test_calculate_features(test_data):
    """Test feature calculation."""
    features = calculate_features(test_data)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] > 0

def test_calculate_regime_features(test_data):
    """Test regime feature calculation."""
    features = calculate_regime_features(test_data)
    assert isinstance(features, np.ndarray)
    assert len(features) > 0

def test_calculate_shrinkage_covariance(test_returns):
    """Test covariance calculation with shrinkage."""
    cov_matrix = calculate_shrinkage_covariance(test_returns, 0.5)
    assert isinstance(cov_matrix, np.ndarray)
    assert cov_matrix.shape == (len(test_returns.columns), len(test_returns.columns))

def test_calculate_risk_parity_weights(test_returns):
    """Test risk parity weight calculation."""
    cov_matrix = calculate_shrinkage_covariance(test_returns, 0.5)
    weights = calculate_risk_parity_weights(cov_matrix)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(test_returns.columns)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)

def test_calculate_mpt_weights(test_returns):
    """Test MPT weight calculation."""
    cov_matrix = calculate_shrinkage_covariance(test_returns, 0.5)
    weights = calculate_mpt_weights(cov_matrix)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(test_returns.columns)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)

def test_apply_portfolio_constraints():
    """Test portfolio constraint application."""
    weights = np.array([0.6, 0.3, 0.1])
    constrained_weights = apply_portfolio_constraints(weights)
    assert isinstance(constrained_weights, np.ndarray)
    assert len(constrained_weights) == len(weights)
    assert np.isclose(np.sum(constrained_weights), 1.0)
    assert np.all(constrained_weights >= 0.01)
    assert np.all(constrained_weights <= 0.5)

def test_check_missing_values(test_data):
    """Test missing value check."""
    result = check_missing_values(test_data, 0.05)
    assert isinstance(result, bool)

def test_check_outliers(test_data):
    """Test outlier check."""
    result = check_outliers(test_data)
    assert isinstance(result, bool)

def test_check_data_consistency(test_data):
    """Test data consistency check."""
    result = check_data_consistency(test_data)
    assert isinstance(result, bool)

def test_generate_warnings():
    """Test warning generation."""
    results = {
        'missing': False,
        'outliers': True,
        'consistency': False
    }
    warnings = generate_warnings(results)
    assert isinstance(warnings, list)
    assert len(warnings) > 0

def test_prepare_training_data(test_data):
    """Test training data preparation."""
    X, y = prepare_training_data(test_data, 60)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]

def test_save_load_models(features, test_data):
    """Test model saving and loading."""
    # Train models
    features.train_models(test_data)
    
    # Save models
    save_models(features.models, features.config.ml_model_dir)
    
    # Load models
    loaded_models = load_models(features.config.ml_model_dir)
    
    assert 'price_prediction' in loaded_models
    assert 'regime_classification' in loaded_models

def test_cleanup(features):
    """Test cleanup."""
    features.cleanup()
    
    # Check if directories are cleaned up
    assert not os.path.exists(features.config.ml_model_dir)
    assert not os.path.exists(features.config.backtest_dir)

@pytest.fixture(autouse=True)
def cleanup_directories():
    """Clean up test directories before and after tests."""
    # Clean up before tests
    if os.path.exists('test_models'):
        shutil.rmtree('test_models')
    if os.path.exists('test_backtests'):
        shutil.rmtree('test_backtests')
        
    yield
    
    # Clean up after tests
    if os.path.exists('test_models'):
        shutil.rmtree('test_models')
    if os.path.exists('test_backtests'):
        shutil.rmtree('test_backtests') 