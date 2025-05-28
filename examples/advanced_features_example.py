import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from utils.advanced_features import AdvancedFeatures, AdvancedConfig

def download_data(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Download data from Yahoo Finance."""
    try:
        data = yf.download(symbol, period=period, interval='1h')
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def main():
    """Main function to demonstrate advanced features."""
    try:
        # Create configuration
        config = AdvancedConfig(
            ml_model_dir='models',
            backtest_dir='backtests',
            require_2fa=True,
            ip_whitelist=['127.0.0.1'],
            alert_thresholds={
                'max_drawdown': 0.1,
                'min_sharpe': 1.0,
                'max_position_size': 0.2
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
            use_redis=False,
            log_level='INFO',
            enable_profiling=True
        )
        
        # Initialize features
        features = AdvancedFeatures(config)
        
        # Download test data
        print("Downloading data...")
        data = download_data('BTC-USD')
        if data is None:
            return
            
        # Train models
        print("Training models...")
        features.train_models(data)
        
        # Make predictions
        print("Making predictions...")
        prediction = features.predict_price(data)
        print(f"Next price prediction: {prediction:.2f}")
        
        # Detect market regime
        print("Detecting market regime...")
        regime = features.detect_regime(data)
        print(f"Current market regime: {regime}")
        
        # Optimize portfolio
        print("Optimizing portfolio...")
        assets = ['BTC-USD', 'ETH-USD', 'BNB-USD']
        returns_data = {}
        
        for asset in assets:
            asset_data = download_data(asset)
            if asset_data is not None:
                returns_data[asset] = asset_data['Close'].pct_change()
                
        returns = pd.DataFrame(returns_data)
        weights = features.optimize_portfolio(returns)
        
        print("Optimal portfolio weights:")
        for asset, weight in zip(assets, weights):
            print(f"{asset}: {weight:.2%}")
            
        # Run backtest
        print("Running backtest...")
        entry_rules = [
            lambda x: x['close'] > x['close'].rolling(20).mean(),
            lambda x: x['volume'] > x['volume'].rolling(20).mean()
        ]
        
        exit_rules = [
            lambda x: x['close'] < x['close'].rolling(20).mean(),
            lambda x: x['volume'] < x['volume'].rolling(20).mean()
        ]
        
        results = features.run_backtest(
            data=data,
            entry_rules=entry_rules,
            exit_rules=exit_rules
        )
        
        print("Backtest results:")
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print(f"Total P&L: {results['total_pnl']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        # Monitor performance
        print("Monitoring performance...")
        metrics = features.calculate_performance_metrics(
            results['trades'],
            results['equity_curve']
        )
        
        print("Performance metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
        # Validate data
        print("Validating data...")
        validation = features.validate_data(data)
        
        print("Data validation results:")
        print(f"Valid: {validation['valid']}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"- {warning}")
                
        # Clean up
        features.cleanup()
        
    except Exception as e:
        print(f"Error in main: {e}")
        
if __name__ == '__main__':
    main() 