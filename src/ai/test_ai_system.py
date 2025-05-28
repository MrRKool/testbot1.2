import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ai.ai_learning import (
    AILearningModule,
    DataCollector,
    MultiTimeframeAnalyzer,
    AdaptiveRiskManager,
    MarketSentimentAnalyzer
)

class TestAISystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Create test config
        cls.config = {
            'initial_balance': 10000,
            'max_position_size': 0.1,
            'max_leverage': 3.0,
            'risk_free_rate': 0.02,
            'exchange': 'binance',
            'timeframes': ['1m', '5m', '1h', '4h'],
            'symbols': ['BTCUSDT']
        }
        
        # Initialize AI module
        cls.ai_module = AILearningModule(cls.config)
        
    def test_data_collection(self):
        """Test data collection functionality"""
        self.logger.info("Testing data collection...")
        
        # Test historical data collection
        data = self.ai_module.data_collector.collect_historical_data('BTCUSDT', '1h', 100)
        self.assertFalse(data.empty, "Historical data collection failed")
        self.assertGreater(len(data), 0, "No data points collected")
        
        # Verify data structure
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        for col in required_columns:
            self.assertIn(col, data.columns, f"Missing column: {col}")
            
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis"""
        self.logger.info("Testing multi-timeframe analysis...")
        
        # Run analysis
        results = self.ai_module.analyze_multi_timeframe('BTCUSDT', years=1)
        
        # Verify results
        self.assertIn('analysis_results', results, "Missing analysis results")
        self.assertIn('optimal_combination', results, "Missing optimal combination")
        
        # Check timeframe relationships
        self.assertIn('correlations', results['analysis_results'], "Missing correlations")
        self.assertIn('trend_alignment', results['analysis_results'], "Missing trend alignment")
        
    def test_risk_management(self):
        """Test risk management system"""
        self.logger.info("Testing risk management...")
        
        # Test risk parameter updates
        trade_outcome = {
            'pnl': 0.02,
            'drawdown': -0.01,
            'position_size': 0.05
        }
        
        market_data = {
            'volatility': 0.015,
            'trend_strength': 0.6,
            'regime': 'trending'
        }
        
        # Update risk parameters
        self.ai_module.risk_manager.update_risk_parameters(trade_outcome, market_data)
        
        # Get current parameters
        risk_params = self.ai_module.risk_manager.get_risk_parameters()
        
        # Verify parameters
        self.assertIn('max_position_size', risk_params, "Missing position size parameter")
        self.assertIn('stop_loss', risk_params, "Missing stop loss parameter")
        self.assertIn('take_profit', risk_params, "Missing take profit parameter")
        
        # Verify parameter bounds
        self.assertGreater(risk_params['max_position_size'], 0, "Invalid position size")
        self.assertLess(risk_params['max_position_size'], 0.2, "Position size too large")
        
    def test_market_sentiment(self):
        """Test market sentiment analysis"""
        self.logger.info("Testing market sentiment analysis...")
        
        # Test sentiment analysis
        sentiment = self.ai_module.sentiment_analyzer.analyze_news('BTCUSDT')
        
        # Verify sentiment score
        self.assertIsInstance(sentiment, float, "Invalid sentiment score type")
        self.assertGreaterEqual(sentiment, -1, "Sentiment score too low")
        self.assertLessEqual(sentiment, 1, "Sentiment score too high")
        
    def test_prediction_system(self):
        """Test prediction system"""
        self.logger.info("Testing prediction system...")
        
        # Create test market data
        market_data = {
            'symbol': 'BTCUSDT',
            'open': 50000,
            'high': 51000,
            'low': 49000,
            'close': 50500,
            'volume': 1000,
            'rsi': 55,
            'macd': 100,
            'macd_signal': 90,
            'bb_upper': 52000,
            'bb_lower': 48000,
            'ema_short': 50300,
            'ema_medium': 50100,
            'ema_long': 50000
        }
        
        # Get prediction
        prediction = self.ai_module.predict(market_data)
        
        # Verify prediction structure
        required_keys = ['position_size', 'stop_loss', 'take_profit', 'confidence']
        for key in required_keys:
            self.assertIn(key, prediction, f"Missing prediction key: {key}")
            
        # Verify prediction bounds
        self.assertGreaterEqual(prediction['position_size'], 0, "Invalid position size")
        self.assertLessEqual(prediction['position_size'], 0.2, "Position size too large")
        self.assertGreater(prediction['stop_loss'], 0, "Invalid stop loss")
        self.assertGreater(prediction['take_profit'], prediction['stop_loss'], "Take profit must be greater than stop loss")
        
    def test_self_training(self):
        """Test self-training functionality"""
        self.logger.info("Testing self-training...")
        
        # Run self-training
        self.ai_module.self_train(['BTCUSDT'])
        
        # Verify training results
        training_status = self.ai_module.get_training_status()
        
        self.assertIn('data_points', training_status, "Missing data points count")
        self.assertIn('performance_metrics', training_status, "Missing performance metrics")
        
        # Verify performance metrics
        metrics = training_status['performance_metrics']
        if metrics:
            self.assertIn('win_rate', metrics, "Missing win rate")
            self.assertIn('avg_profit', metrics, "Missing average profit")
            self.assertIn('sharpe_ratio', metrics, "Missing Sharpe ratio")
            
    def test_system_health(self):
        """Test system health monitoring"""
        self.logger.info("Testing system health...")
        
        # Get system health
        health = self.ai_module.get_system_health()
        
        # Verify health check results
        self.assertIn('overall_health', health, "Missing overall health status")
        self.assertIn('components', health, "Missing component health checks")
        
        # Check component health
        for component, status in health['components'].items():
            self.assertIn('status', status, f"Missing status for {component}")
            self.assertIn('message', status, f"Missing message for {component}")
            
    def test_state_management(self):
        """Test state saving and loading"""
        self.logger.info("Testing state management...")
        
        # Save state
        self.ai_module.save_state()
        
        # Verify saved files
        required_files = [
            'models/ai_model',
            'models/scaler.pkl',
            'models/performance_history.csv',
            'models/risk_manager_state.json'
        ]
        
        for file in required_files:
            self.assertTrue(os.path.exists(file), f"Missing saved file: {file}")
            
        # Create new instance and load state
        new_ai_module = AILearningModule(self.config)
        new_ai_module.load_state()
        
        # Verify loaded state
        self.assertGreater(len(new_ai_module.feature_data), 0, "No feature data loaded")
        self.assertGreater(len(new_ai_module.performance_history), 0, "No performance history loaded")
        
if __name__ == '__main__':
    unittest.main(verbosity=2) 