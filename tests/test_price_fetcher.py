import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Voeg de root directory toe aan de Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trading.price_fetcher import PriceFetcher

class TestPriceFetcher(unittest.TestCase):
    """Test suite voor de PriceFetcher class."""

    def setUp(self):
        """Setup voor elke test."""
        self.config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'rate_limit': {
                'calls_per_second': 2,
                'calls_per_minute': 50,
                'calls_per_hour': 1000
            },
            'cache_timeout': 60
        }
        self.fetcher = PriceFetcher(self.config)

    def test_initialization(self):
        """Test de initialisatie van de PriceFetcher."""
        self.assertEqual(self.fetcher.api_key, 'test_key')
        self.assertEqual(self.fetcher.api_secret, 'test_secret')
        self.assertEqual(self.fetcher.cache_timeout, 60)
        self.assertIsNotNone(self.fetcher.rate_limiter)
        self.assertIsInstance(self.fetcher.price_cache, dict)

    @patch('utils.price_fetcher.requests.get')
    def test_get_price_success(self, mock_get):
        """Test succesvol ophalen van een prijs."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'price': '50000.0'}
        mock_get.return_value = mock_response

        # Test
        price = self.fetcher.get_price('BTCUSDT')
        self.assertEqual(price, 50000.0)
        self.assertIn('BTCUSDT', self.fetcher.price_cache)

    @patch('utils.price_fetcher.requests.get')
    def test_get_price_failure(self, mock_get):
        """Test falen van prijs ophalen."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_get.return_value = mock_response

        # Test
        with self.assertRaises(Exception):
            self.fetcher.get_price('BTCUSDT')

    @patch('utils.price_fetcher.requests.get')
    def test_get_klines_success(self, mock_get):
        """Test succesvol ophalen van klines."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result': {
                'list': [
                    ['1625097600000', '35000.0', '35100.0', '34900.0', '35050.0', '100.0', '3500000.0'],
                    ['1625097900000', '35050.0', '35200.0', '35000.0', '35150.0', '150.0', '5250000.0']
                ]
            }
        }
        mock_get.return_value = mock_response

        # Test
        df = self.fetcher.get_klines('BTCUSDT', '15', 2)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['open', 'high', 'low', 'close', 'volume', 'turnover'])

    @patch('utils.price_fetcher.requests.get')
    def test_get_klines_invalid_data(self, mock_get):
        """Test ophalen van klines met ongeldige data."""
        # Mock response met ontbrekende data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result': {
                'list': [
                    ['1625097600000', '35000.0', '35100.0']  # Incomplete data
                ]
            }
        }
        mock_get.return_value = mock_response

        # Test
        df = self.fetcher.get_klines('BTCUSDT', '15', 1)
        self.assertIsNone(df)

    def test_clear_cache(self):
        """Test het legen van de cache."""
        # Voeg wat test data toe aan de cache
        self.fetcher.price_cache['BTCUSDT'] = {
            'price': 50000.0,
            'timestamp': datetime.now().timestamp()
        }
        
        # Test
        self.fetcher.clear_cache()
        self.assertEqual(len(self.fetcher.price_cache), 0)

    @patch('utils.price_fetcher.requests.get')
    def test_rate_limiting(self, mock_get):
        """Test rate limiting functionaliteit."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'price': '50000.0'}
        mock_get.return_value = mock_response

        # Maak meerdere requests
        for _ in range(3):
            self.fetcher.get_price('BTCUSDT')

        # Controleer of de rate limiter correct werkt
        self.assertLessEqual(
            self.fetcher.rate_limiter.get_current_usage()['calls_per_second'],
            self.config['rate_limit']['calls_per_second']
        )

    def test_generate_signature(self):
        """Test het genereren van de API signature."""
        params = {
            'symbol': 'BTCUSDT',
            'timestamp': '1625097600000'
        }
        signature = self.fetcher._generate_signature(params)
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 0)

    @patch('utils.price_fetcher.requests.get')
    def test_retry_mechanism(self, mock_get):
        """Test het retry mechanisme bij fouten."""
        # Mock response met eerst een fout, dan succes
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'price': '50000.0'}
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]

        # Test
        price = self.fetcher.get_price('BTCUSDT')
        self.assertEqual(price, 50000.0)
        self.assertEqual(mock_get.call_count, 2)

if __name__ == '__main__':
    unittest.main() 