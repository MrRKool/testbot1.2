import unittest
import logging
import pandas as pd
import numpy as np
from ai_learning import (
    AILearningModule,
    DataCollector,
    MarketSentimentAnalyzer,
    EnsembleModel,
    AdaptiveRiskManager,
    MultiTimeframeAnalyzer,
    VolatilityAnalyzer
)

class TestComprehensive(unittest.TestCase):
    def setUp(self):
        self.config = {
            'initial_balance': 10000,
            'max_position_size': 0.1,
            'max_leverage': 3.0,
            'risk_free_rate': 0.02,
            'exchange': 'binance',
            'timeframes': ['1h', '4h'],
            'symbols': ['BTCUSDT'],
            'feature_dim': 50,
            'sequence_length': 60,
            'model_dir': 'models',
            'data_dir': 'data',
            'cache_dir': 'cache'
        }
        
    def test_data_collection(self):
        """Test data verzameling en preprocessing"""
        collector = DataCollector(self.config)
        
        # Test historische data verzameling
        data = collector.collect_historical_data('BTCUSDT', '1h', 100)
        self.assertFalse(data.empty, "Data verzameling mislukt")
        self.assertTrue('close' in data.columns, "Close kolom ontbreekt")
        self.assertTrue('volume' in data.columns, "Volume kolom ontbreekt")
        
        # Test technische indicators
        self.assertTrue('rsi' in data.columns, "RSI indicator ontbreekt")
        self.assertTrue('macd' in data.columns, "MACD indicator ontbreekt")
        self.assertTrue('bb_upper' in data.columns, "Bollinger Bands ontbreken")
        
    def test_sentiment_analysis(self):
        """Test sentiment analyse functionaliteit"""
        analyzer = MarketSentimentAnalyzer(self.config)
        
        # Test nieuws sentiment
        news_sentiment = analyzer.analyze_news('BTCUSDT')
        self.assertIsInstance(news_sentiment, float, "Nieuws sentiment moet een float zijn")
        self.assertTrue(-1 <= news_sentiment <= 1, "Sentiment moet tussen -1 en 1 liggen")
        
        # Test technische sentiment
        technical_sentiment = analyzer._get_technical_sentiment()
        self.assertIsInstance(technical_sentiment, float, "Technisch sentiment moet een float zijn")
        
    def test_ensemble_model(self):
        """Test ensemble model functionaliteit"""
        # Zet feature_dim en sequence_length op het hoogste niveau
        config = {
            'feature_dim': 50,
            'sequence_length': 60,
            'model': {
                'ensemble': {
                    'models': ['lstm', 'gru', 'transformer'],
                    'weights': [0.4, 0.3, 0.3]
                }
            },
            'data': {
                'timeframes': ['1m', '5m', '1h', '4h'],
                'features': ['price', 'volume', 'technical', 'sentiment', 'volatility']
            }
        }
        
        # Initialize analyzers
        sentiment_analyzer = MarketSentimentAnalyzer(config)
        volatility_analyzer = VolatilityAnalyzer(config)
        
        # Initialize ensemble model with analyzers
        ensemble = EnsembleModel(config, sentiment_analyzer=sentiment_analyzer, volatility_analyzer=volatility_analyzer)
        
        # Test data preparation
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(100, 10, 100)
        })
        test_data.set_index('timestamp', inplace=True)
        
        # Test feature preparation
        features = ensemble.prepare_features(test_data)
        assert isinstance(features, np.ndarray), "Features moeten een numpy array zijn"
        
        # Test prediction
        prediction = ensemble.predict(features)
        self.assertIsInstance(prediction, np.ndarray, "Prediction moet een numpy array zijn")
        
    def test_risk_management(self):
        """Test risk management functionaliteit"""
        risk_manager = AdaptiveRiskManager(self.config)
        
        # Test initiële parameters
        initial_params = risk_manager.get_risk_parameters()
        self.assertIn('max_position_size', initial_params)
        self.assertIn('stop_loss', initial_params)
        self.assertIn('take_profit', initial_params)
        
        # Test parameter updates
        trade_outcome = {
            'pnl': 0.02,
            'drawdown': -0.01,
            'position_size': 0.05
        }
        market_data = {
            'regime': 'trending',
            'volatility': 0.02
        }
        risk_manager.update_risk_parameters(trade_outcome, market_data)
        updated_params = risk_manager.get_risk_parameters()
        self.assertNotEqual(initial_params, updated_params, "Parameters moeten zijn bijgewerkt")
        
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analyse"""
        analyzer = MultiTimeframeAnalyzer(self.config)
        
        # Test data verzameling voor meerdere timeframes
        data = analyzer.collect_historical_data('BTCUSDT', years=1)
        self.assertGreater(len(data), 0, "Geen data verzameld voor timeframes")
        
        # Test timeframe analyse
        analysis = analyzer.analyze_timeframe_relationships()
        self.assertIn('correlations', analysis)
        self.assertIn('trend_alignment', analysis)
        
    def test_ai_module_integration(self):
        """Test integratie van alle componenten"""
        ai = AILearningModule(self.config)
        
        # Test component initialisatie
        self.assertIsNotNone(ai.data_collector, "Data collector niet geïnitialiseerd")
        self.assertIsNotNone(ai.ensemble, "Ensemble model niet geïnitialiseerd")
        self.assertIsNotNone(ai.risk_manager, "Risk manager niet geïnitialiseerd")
        self.assertIsNotNone(ai.sentiment_analyzer, "Sentiment analyzer niet geïnitialiseerd")
        
        # Test voorspelling
        market_data = {
            'symbol': 'BTCUSDT',
            'close': 50000,
            'volume': 1000,
            'timestamp': pd.Timestamp.now()
        }
        prediction = ai.predict(market_data)
        self.assertIn('position_size', prediction)
        self.assertIn('stop_loss', prediction)
        self.assertIn('take_profit', prediction)
        self.assertIn('confidence', prediction)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main() 