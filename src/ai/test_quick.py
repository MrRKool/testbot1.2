import unittest
import logging
from ai_learning import AILearningModule, DataCollector, MarketSentimentAnalyzer

class TestQuick(unittest.TestCase):
    def setUp(self):
        self.config = {
            'initial_balance': 10000,
            'max_position_size': 0.1,
            'max_leverage': 3.0,
            'risk_free_rate': 0.02,
            'exchange': 'binance',
            'timeframes': ['1h'],  # Alleen 1h voor snelle test
            'symbols': ['BTCUSDT'],
            'feature_dim': 50,
            'sequence_length': 60,
            'model_dir': 'models',
            'data_dir': 'data',
            'cache_dir': 'cache'
        }
        
    def test_data_collection(self):
        """Test data verzameling"""
        collector = DataCollector(self.config)
        data = collector.collect_historical_data('BTCUSDT', '1h', 100)
        self.assertFalse(data.empty, "Data verzameling mislukt")
        self.assertTrue('close' in data.columns, "Close kolom ontbreekt")
        
    def test_sentiment_analysis(self):
        """Test sentiment analyse"""
        analyzer = MarketSentimentAnalyzer(self.config)
        sentiment = analyzer.analyze_news('BTCUSDT')
        self.assertIsInstance(sentiment, float, "Sentiment moet een float zijn")
        
    def test_ai_module_initialization(self):
        """Test AI module initialisatie"""
        ai = AILearningModule(self.config)
        self.assertIsNotNone(ai.data_collector, "Data collector niet geïnitialiseerd")
        self.assertIsNotNone(ai.ensemble, "Ensemble model niet geïnitialiseerd")
        self.assertIsNotNone(ai.risk_manager, "Risk manager niet geïnitialiseerd")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main() 