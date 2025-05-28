import logging
from datetime import datetime
import pandas as pd
from utils.advanced_features import AdvancedFeatures, AdvancedConfig
from utils.config import Config
from utils.risk_manager import RiskManager
from utils.data.data_manager import DataManager
from utils.monitoring.performance_monitor import PerformanceMonitor

def integrate_advanced_features(config: Config):
    """Integrate advanced features with the trading bot."""
    try:
        # Set up logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create advanced config
        advanced_config = AdvancedConfig(
            ml_model_dir='models',
            backtest_dir='backtests',
            require_2fa=config.security.require_2fa,
            ip_whitelist=config.security.ip_whitelist,
            alert_thresholds={
                'max_drawdown': config.risk.max_drawdown,
                'min_sharpe': config.risk.min_sharpe,
                'max_position_size': config.risk.max_position_size
            },
            training_window=60,
            batch_size=32,
            use_gpu=True,
            risk_parity=True,
            shrinkage_factor=0.5,
            parallel_processing=True,
            num_workers=4,
            async_monitoring=True,
            metrics_aggregation=True,
            parallel_validation=True,
            cache_enabled=True,
            use_redis=config.cache.use_redis,
            log_level=config.logging.level,
            enable_profiling=config.monitoring.enable_profiling
        )
        
        # Initialize components
        features = AdvancedFeatures(advanced_config)
        risk_manager = RiskManager(config.risk)
        data_manager = DataManager(config.data)
        performance_monitor = PerformanceMonitor(config.monitoring)
        
        # Train models with historical data
        logger.info("Training models with historical data...")
        historical_data = data_manager.get_historical_data()
        features.train_models(historical_data)
        
        # Start real-time monitoring loop
        logger.info("Starting real-time monitoring loop...")
        while True:
            try:
                # Update current data
                current_data = data_manager.get_current_data()
                
                # Make predictions
                prediction = features.predict_price(current_data)
                
                # Detect market regime
                regime = features.detect_regime(current_data)
                
                # Update risk manager with predictions and regime
                risk_manager.update(
                    prediction=prediction,
                    market_regime=regime
                )
                
                # Optimize portfolio if enabled
                if config.portfolio.optimize:
                    returns = data_manager.get_returns()
                    weights = features.optimize_portfolio(returns)
                    risk_manager.update_weights(weights)
                    
                # Monitor performance
                performance_monitor.update(
                    trades=data_manager.get_trades(),
                    equity_curve=data_manager.get_equity_curve()
                )
                
                # Validate data quality
                validation = features.validate_data(current_data)
                if not validation['valid']:
                    logger.warning(f"Data quality issues: {validation['warnings']}")
                    # Clean data if needed
                    current_data = features.clean_data(current_data)
                    
                # Check security for API requests
                if config.security.enabled:
                    is_valid = features.validate_api_request({
                        'ip': config.security.current_ip,
                        'endpoint': config.security.current_endpoint,
                        'method': config.security.current_method,
                        '2fa_code': config.security.current_2fa_code
                    })
                    if not is_valid:
                        logger.error("Invalid API request")
                        continue
                        
                # Log status
                logger.info(f"Current regime: {regime}")
                logger.info(f"Predicted price: {prediction:.2f}")
                logger.info(f"Current risk level: {risk_manager.get_risk_level()}")
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in integration: {e}")
        raise
        
if __name__ == '__main__':
    # Load configuration
    config = Config()
    
    # Integrate advanced features
    integrate_advanced_features(config) 