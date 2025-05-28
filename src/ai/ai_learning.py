import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import os
import gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import optuna
from optuna.visualization import plot_optimization_history
from transformers import AutoTokenizer, AutoModel
import torch
from textblob import TextBlob
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import ccxt
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline
import random
import itertools
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for Reinforcement Learning"""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),  # [position_size, stop_loss, take_profit]
            high=np.array([1, 0.1, 0.2]),  # Max 100% position, 10% SL, 20% TP
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(50,),  # Market features
            dtype=np.float32
        )
        self.current_step = 0
        self.data = None
        self.position = None
        self.balance = config.get('initial_balance', 10000)
        self.initial_balance = self.balance
        
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.position = None
        self.balance = self.initial_balance
        return self._get_observation()
        
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Extract action components
        position_size, stop_loss, take_profit = action
        
        # Get current market state
        current_price = self.data['close'].iloc[self.current_step]
        
        # Execute trade if no position
        if self.position is None:
            self.position = {
                'size': position_size * self.balance / current_price,
                'entry_price': current_price,
                'stop_loss': current_price * (1 - stop_loss),
                'take_profit': current_price * (1 + take_profit)
            }
        
        # Check if position should be closed
        reward = 0
        done = False
        
        if self.position:
            # Check stop loss
            if current_price <= self.position['stop_loss']:
                reward = -stop_loss
                self._close_position(current_price)
                done = True
            # Check take profit
            elif current_price >= self.position['take_profit']:
                reward = take_profit
                self._close_position(current_price)
                done = True
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step
        }
        
        return observation, reward, done, info
        
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= len(self.data):
            return np.zeros(50)
            
        current_data = self.data.iloc[self.current_step]
        
        # Combine all features
        features = np.concatenate([
            current_data[['open', 'high', 'low', 'close', 'volume']].values,
            current_data[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']].values,
            current_data[['ema_short', 'ema_medium', 'ema_long']].values,
            [self.balance / self.initial_balance]  # Normalized balance
        ])
        
        return features
        
    def _close_position(self, current_price):
        """Close current position"""
        if self.position:
            pnl = (current_price - self.position['entry_price']) * self.position['size']
            self.balance += pnl
            self.position = None

class MarketSentimentAnalyzer:
    """Analyzes market sentiment from various sources"""
    def __init__(self, config: dict):
        self.config = config
        # Set environment variable to disable parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
        self.sentiment_cache = {}
        self.logger = logging.getLogger(__name__)
        self.data_collector = DataCollector(config)
        
    def _get_price_data(self) -> pd.DataFrame:
        """Get historical price data"""
        try:
            return self.data_collector.collect_historical_data('BTCUSDT', '1h', 100)
        except Exception as e:
            self.logger.error(f"Error getting price data: {str(e)}")
            return pd.DataFrame()
            
    def _get_volume_data(self) -> pd.Series:
        """Get historical volume data"""
        try:
            data = self._get_price_data()
            return data['volume'] if not data.empty else pd.Series()
        except Exception as e:
            self.logger.error(f"Error getting volume data: {str(e)}")
            return pd.Series()
            
    def _get_current_price(self) -> float:
        """Get current price"""
        try:
            data = self._get_price_data()
            return data['close'].iloc[-1] if not data.empty else 0.0
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return 0.0

    def analyze_news(self, symbol: str) -> float:
        """Analyze news sentiment for a symbol"""
        try:
            if symbol in self.sentiment_cache:
                return self.sentiment_cache[symbol]
            news = self._get_news(symbol)
            if not news:
                return 0.0
            sentiments = []
            for article in news:
                text = article['title'] + " " + (article.get('description', ''))
                result = self.sentiment_analyzer(text)[0]
                sentiment = 1.0 if result['label'] == 'POSITIVE' else -1.0
                sentiment *= result['score']
                sentiments.append(sentiment)
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            self.sentiment_cache[symbol] = avg_sentiment
            return avg_sentiment
        except Exception as e:
            logging.error(f"Error analyzing news sentiment: {str(e)}")
            return 0.0

    def _get_news(self, symbol: str) -> List[Dict]:
        """Get news from various sources"""
        news = []
        try:
            # Get news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news.extend(ticker.news)
            
            # Get news from CryptoCompare
            if symbol.endswith('USDT'):
                crypto_symbol = symbol[:-4]
                response = requests.get(f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key=YOUR_API_KEY")
                if response.status_code == 200:
                    crypto_news = response.json()['Data']
                    news.extend(crypto_news)
                    
        except Exception as e:
            logging.error(f"Error fetching news: {str(e)}")
            
        return news

    def _get_news_sentiment(self):
        """Get news sentiment from various sources"""
        try:
            # Get news from various sources
            news_sources = [
                'coindesk.com',
                'cointelegraph.com',
                'bitcoin.com',
                'newsbtc.com'
            ]
            
            sentiments = []
            for source in news_sources:
                try:
                    # Get latest news
                    news = self._fetch_news(source)
                    
                    # Analyze sentiment
                    for article in news:
                        sentiment = TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity
                        sentiments.append(sentiment)
                except Exception as e:
                    logger.warning(f"Error fetching news from {source}: {str(e)}")
                    
            # Return average sentiment
            return np.mean(sentiments) if sentiments else 0
            
        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {str(e)}")
            return 0
            
    def _get_social_sentiment(self):
        """Get social media sentiment"""
        try:
            # Get social media data
            social_sources = [
                'twitter',
                'reddit',
                'telegram'
            ]
            
            sentiments = []
            for source in social_sources:
                try:
                    # Get latest posts
                    posts = self._fetch_social_posts(source)
                    
                    # Analyze sentiment
                    for post in posts:
                        sentiment = TextBlob(post['text']).sentiment.polarity
                        sentiments.append(sentiment)
                except Exception as e:
                    logger.warning(f"Error fetching posts from {source}: {str(e)}")
                    
            # Return average sentiment
            return np.mean(sentiments) if sentiments else 0
            
        except Exception as e:
            logger.error(f"Error in social sentiment analysis: {str(e)}")
            return 0
            
    def _get_market_sentiment(self):
        """Get overall market sentiment"""
        try:
            # Combine different sentiment sources
            news_sentiment = self._get_news_sentiment()
            social_sentiment = self._get_social_sentiment()
            
            # Technical sentiment (based on indicators)
            technical_sentiment = self._get_technical_sentiment()
            
            # Weighted average
            weights = {
                'news': 0.3,
                'social': 0.3,
                'technical': 0.4
            }
            
            market_sentiment = (
                news_sentiment * weights['news'] +
                social_sentiment * weights['social'] +
                technical_sentiment * weights['technical']
            )
            
            return market_sentiment
            
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {str(e)}")
            return 0
            
    def _get_technical_sentiment(self):
        """Get sentiment based on technical indicators"""
        try:
            # Calculate technical indicators
            indicators = {
                'rsi': self._calculate_rsi(),
                'macd': self._calculate_macd(),
                'bb': self._calculate_bollinger_bands(),
                'volume': self._calculate_volume_profile()
            }
            
            # Convert indicators to sentiment scores
            sentiment_scores = {
                'rsi': self._rsi_to_sentiment(indicators['rsi']),
                'macd': self._macd_to_sentiment(indicators['macd']),
                'bb': self._bb_to_sentiment(indicators['bb']),
                'volume': self._volume_to_sentiment(indicators['volume'])
            }
            
            # Weighted average
            weights = {
                'rsi': 0.3,
                'macd': 0.3,
                'bb': 0.2,
                'volume': 0.2
            }
            
            technical_sentiment = sum(
                score * weights[indicator]
                for indicator, score in sentiment_scores.items()
            )
            
            return technical_sentiment
            
        except Exception as e:
            logger.error(f"Error in technical sentiment analysis: {str(e)}")
            return 0
            
    def _get_implied_volatility(self):
        """Get implied volatility from options data"""
        try:
            # Get options data
            options_data = self._fetch_options_data()
            
            if not options_data:
                return None
                
            # Calculate implied volatility
            iv = self._calculate_implied_volatility(options_data)
            
            return iv
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {str(e)}")
            return None
            
    def _classify_volatility_regime(self, historical_vol):
        """Classify current volatility regime"""
        try:
            # Calculate volatility percentiles
            vol_percentiles = np.percentile(historical_vol, [25, 50, 75])
            
            # Classify regime
            current_vol = historical_vol.iloc[-1]
            
            if current_vol < vol_percentiles[0]:
                return 'low'
            elif current_vol < vol_percentiles[1]:
                return 'medium_low'
            elif current_vol < vol_percentiles[2]:
                return 'medium_high'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {str(e)}")
            return 'medium'

    def _calculate_rsi(self, period=14):
        """Calculate RSI indicator"""
        try:
            # Get price data
            prices = self._get_price_data()
            
            # Calculate RSI
            delta = prices['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50
            
    def _calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            # Get price data
            prices = self._get_price_data()
            
            # Calculate MACD
            exp1 = prices['close'].ewm(span=fast, adjust=False).mean()
            exp2 = prices['close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': macd.iloc[-1] - signal_line.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
    def _calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            # Get price data
            prices = self._get_price_data()
            
            # Calculate Bollinger Bands
            middle_band = prices['close'].rolling(window=period).mean()
            std = prices['close'].rolling(window=period).std()
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return {
                'upper': upper_band.iloc[-1],
                'middle': middle_band.iloc[-1],
                'lower': lower_band.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {'upper': 0, 'middle': 0, 'lower': 0}
            
    def _calculate_volume_profile(self, period=20):
        """Calculate volume profile"""
        try:
            # Get volume data
            volume = self._get_volume_data()
            
            # Calculate volume profile
            avg_volume = volume.rolling(window=period).mean()
            volume_std = volume.rolling(window=period).std()
            
            return {
                'current': volume.iloc[-1],
                'average': avg_volume.iloc[-1],
                'std': volume_std.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {'current': 0, 'average': 0, 'std': 0}
            
    def _rsi_to_sentiment(self, rsi):
        """Convert RSI to sentiment score (-1 to 1)"""
        try:
            # RSI ranges:
            # 0-20: Extremely oversold (bullish)
            # 20-40: Oversold (slightly bullish)
            # 40-60: Neutral
            # 60-80: Overbought (slightly bearish)
            # 80-100: Extremely overbought (bearish)
            
            if rsi <= 20:
                return 1.0  # Extremely bullish
            elif rsi <= 40:
                return 0.5  # Slightly bullish
            elif rsi <= 60:
                return 0.0  # Neutral
            elif rsi <= 80:
                return -0.5  # Slightly bearish
            else:
                return -1.0  # Extremely bearish
                
        except Exception as e:
            logger.error(f"Error converting RSI to sentiment: {str(e)}")
            return 0
            
    def _macd_to_sentiment(self, macd_data):
        """Convert MACD to sentiment score (-1 to 1)"""
        try:
            macd = macd_data['macd']
            signal = macd_data['signal']
            histogram = macd_data['histogram']
            
            # Calculate sentiment based on MACD components
            if histogram > 0:
                # Bullish
                return min(histogram / signal, 1.0) if signal > 0 else 0.5
            else:
                # Bearish
                return max(histogram / signal, -1.0) if signal < 0 else -0.5
                
        except Exception as e:
            logger.error(f"Error converting MACD to sentiment: {str(e)}")
            return 0
            
    def _bb_to_sentiment(self, bb_data):
        """Convert Bollinger Bands to sentiment score (-1 to 1)"""
        try:
            current_price = self._get_current_price()
            upper = bb_data['upper']
            lower = bb_data['lower']
            middle = bb_data['middle']
            
            # Calculate position relative to bands
            if current_price >= upper:
                return -1.0  # Extremely bearish
            elif current_price <= lower:
                return 1.0  # Extremely bullish
            else:
                # Linear interpolation between bands
                return 2 * (current_price - middle) / (upper - lower)
                
        except Exception as e:
            logger.error(f"Error converting Bollinger Bands to sentiment: {str(e)}")
            return 0
            
    def _volume_to_sentiment(self, volume_data):
        """Convert volume profile to sentiment score (-1 to 1)"""
        try:
            current = volume_data['current']
            average = volume_data['average']
            std = volume_data['std']
            
            # Calculate volume ratio
            volume_ratio = (current - average) / std
            
            # Convert to sentiment
            if volume_ratio > 2:
                return 1.0  # Extremely high volume (bullish)
            elif volume_ratio > 1:
                return 0.5  # High volume (slightly bullish)
            elif volume_ratio < -2:
                return -1.0  # Extremely low volume (bearish)
            elif volume_ratio < -1:
                return -0.5  # Low volume (slightly bearish)
            else:
                return 0.0  # Normal volume (neutral)
                
        except Exception as e:
            logger.error(f"Error converting volume to sentiment: {str(e)}")
            return 0

    def _get_price_sentiment(self, data: pd.DataFrame) -> float:
        """Analyze price action sentiment"""
        try:
            if data.empty:
                return 0.0
                
            # Calculate price momentum
            returns = data['close'].pct_change()
            momentum = returns.mean()
            
            # Calculate trend strength
            ema_short = data['ema_short'].iloc[-1]
            ema_long = data['ema_long'].iloc[-1]
            trend_strength = (ema_short - ema_long) / ema_long
            
            # Calculate support/resistance levels
            price = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            
            # Normalize price position between bands
            price_position = (price - bb_lower) / (bb_upper - bb_lower)
            
            # Combine signals
            sentiment = (
                0.4 * np.sign(momentum) +
                0.4 * np.sign(trend_strength) +
                0.2 * (price_position - 0.5) * 2
            )
            
            return float(np.clip(sentiment, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error in price sentiment analysis: {str(e)}")
            return 0.0
            
    def _get_volume_sentiment(self, data: pd.DataFrame) -> float:
        """Analyze volume sentiment"""
        try:
            if data.empty:
                return 0.0
                
            # Calculate volume trend
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_trend = (data['volume'].iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
            
            # Calculate volume-price relationship
            price_change = data['close'].pct_change()
            volume_price_corr = price_change.corr(data['volume'])
            
            # Calculate volume momentum
            volume_momentum = data['volume'].pct_change().mean()
            
            # Combine signals
            sentiment = (
                0.4 * np.sign(volume_trend) +
                0.3 * np.sign(volume_price_corr) +
                0.3 * np.sign(volume_momentum)
            )
            
            return float(np.clip(sentiment, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error in volume sentiment analysis: {str(e)}")
            return 0.0

class VolatilityAnalyzer:
    """Analyzes market volatility and related metrics"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_collector = DataCollector(config)
        
    def get_implied_volatility(self, symbol: str = 'BTCUSDT') -> float:
        """Get implied volatility for a symbol"""
        try:
            # Placeholder for actual implied volatility calculation
            # In a real implementation, this would fetch options data and calculate IV
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.0
            
    def classify_volatility_regime(self, historical_vol: pd.Series) -> float:
        """Classify current volatility regime"""
        try:
            if historical_vol.empty:
                return 0.0
                
            # Calculate volatility percentiles
            vol_percentiles = np.percentile(historical_vol.dropna(), [25, 50, 75])
            
            # Classify regime
            current_vol = historical_vol.iloc[-1]
            
            if current_vol < vol_percentiles[0]:
                return 0.25  # low
            elif current_vol < vol_percentiles[1]:
                return 0.5   # medium_low
            elif current_vol < vol_percentiles[2]:
                return 0.75  # medium_high
            else:
                return 1.0   # high
                
        except Exception as e:
            self.logger.error(f"Error classifying volatility regime: {str(e)}")
            return 0.0
            
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        try:
            df = pd.DataFrame(data)
            
            # Historical volatility
            df['hist_vol'] = df['close'].pct_change().rolling(window=20).std()
            
            # Volatility regime (as float)
            df['vol_regime'] = self.classify_volatility_regime(df['hist_vol'])
            
            # Implied volatility (placeholder)
            df['implied_vol'] = self.get_implied_volatility()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return pd.DataFrame()

class EnsembleModel:
    def __init__(self, config: Dict[str, Any], sentiment_analyzer=None, volatility_analyzer=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.volatility_analyzer = volatility_analyzer
        self.models = []
        self.model_types = {}
        self.sequence_length = config.get('sequence_length', 60)
        self.feature_dim = config.get('feature_dim', 42)  # Default to 42 if not specified
        self.performance_metrics = {}  # Initialize performance_metrics
        self._keras_predict_fn = None  # Store tf.function outside loop
        self.initialize_models()
        
    def _create_keras_predict_fn(self, model):
        # Only create once per model
        @tf.function(reduce_retracing=True)
        def keras_predict(X):
            # Ensure input shape is correct
            if len(X.shape) == 3:
                X = tf.reshape(X, (-1, X.shape[-1]))
            return model(X, training=False)
        return keras_predict
        
    def prepare_features(self, data):
        """Prepare and engineer features for all models"""
        try:
            features = {}
            # Technical indicators
            features['technical'] = self._calculate_technical_indicators(data)
            # Market microstructure features
            features['microstructure'] = self._calculate_microstructure_features(data)
            # Sentiment features
            if self.sentiment_analyzer is not None:
                features['sentiment'] = self._calculate_sentiment_features(data)
            else:
                # Fallback: fill with zeros
                df = pd.DataFrame(data)
                features['sentiment'] = pd.DataFrame(np.zeros((len(df), 3)), columns=['news_sentiment','social_sentiment','market_sentiment'])
            # Volatility features
            if self.volatility_analyzer is not None:
                features['volatility'] = self._calculate_volatility_features(data)
            else:
                # Fallback: fill with zeros
                df = pd.DataFrame(data)
                features['volatility'] = pd.DataFrame(np.zeros((len(df), 3)), columns=['hist_vol','vol_regime','implied_vol'])
            # Combine all features
            combined = self._combine_features(features)
            # Update feature dimension if changed
            if self.feature_dim is None or self.feature_dim != combined.shape[1]:
                self.feature_dim = combined.shape[1]
                self.logger.info(f"(Re)setting feature dimension to {self.feature_dim} and reinitializing models")
                self.initialize_models()
            return combined
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None
        
    def initialize_models(self):
        try:
            self.models = []
            self.model_types = {k: None for k in self.model_types}
            dummy_X = np.random.randn(10, self.feature_dim)
            dummy_y = np.random.randint(0, 3, size=(10,))
            dummy_y_onehot = np.eye(3)[dummy_y]
            nn_model = self._create_nn_model()
            nn_model.fit(dummy_X, dummy_y_onehot, epochs=1, verbose=0)
            self.models.append(nn_model)
            self.model_types['nn'] = nn_model
            lstm_model = self._create_lstm_model()
            dummy_X_lstm = np.random.randn(10, self.sequence_length, self.feature_dim)
            lstm_model.fit(dummy_X_lstm, dummy_y_onehot, epochs=1, verbose=0)
            self.models.append(lstm_model)
            self.model_types['lstm'] = lstm_model
            xgb_model = self._create_xgboost_model()
            xgb_model.fit(dummy_X, dummy_y)
            self.models.append(xgb_model)
            self.model_types['xgb'] = xgb_model
            rf_model = self._create_rf_model()
            rf_model.fit(dummy_X, dummy_y)
            self.models.append(rf_model)
            self.model_types['rf'] = rf_model
            # (Re)create keras predict function for the current NN model
            self._keras_predict_fn = self._create_keras_predict_fn(nn_model)
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
        
    def _create_nn_model(self):
        """Create an optimized feedforward neural network for trading"""
        if self.feature_dim is None:
            self.logger.warning("Feature dimension not set, using default from config")
            self.feature_dim = self.config.get('feature_dim', 50)
            
        inputs = layers.Input(shape=(self.feature_dim,))
        x = layers.Dense(128)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(3, activation='softmax')(x)  # Buy, Sell, Hold
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
        
    def _create_lstm_model(self):
        """Create LSTM model for sequence prediction"""
        if self.feature_dim is None:
            self.logger.warning("Feature dimension not set, using default from config")
            self.feature_dim = self.config.get('feature_dim', 50)
            
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def _create_xgboost_model(self):
        """Create XGBoost model for feature importance"""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3
        )
        
    def _create_rf_model(self):
        """Create Random Forest model for robustness"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
        
    def _calculate_technical_indicators(self, data):
        """Calculate advanced technical indicators"""
        df = pd.DataFrame(data)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Trend indicators
        df['ema_short'] = ta.ema(df['close'], length=9)
        df['ema_medium'] = ta.ema(df['close'], length=21)
        df['ema_long'] = ta.ema(df['close'], length=50)
        
        # Momentum indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        # Volume indicators
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # VWAP (only if we have all required columns)
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            try:
                df['vwap'] = ta.vwap(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=df['volume']
                )
            except Exception as e:
                self.logger.warning(f"Could not calculate VWAP: {str(e)}")
                df['vwap'] = df['close']  # Fallback to close price
        
        # Volatility indicators
        try:
            bbands = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
        except Exception as e:
            self.logger.warning(f"Could not calculate Bollinger Bands: {str(e)}")
            # Fallback values
            df['bb_upper'] = df['close'] * 1.02
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close'] * 0.98
            
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        
        return df
        
    def _calculate_microstructure_features(self, data):
        """Calculate market microstructure features"""
        df = pd.DataFrame(data)
        
        # Order flow features
        if 'ask' in df.columns and 'bid' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = df['spread'] / df['bid']
        else:
            df['spread'] = 0.0
            df['spread_pct'] = 0.0
        
        # Volume profile
        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            total_vol = df['buy_volume'] + df['sell_volume']
            df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / total_vol.replace(0, np.nan)
            df['volume_imbalance'] = df['volume_imbalance'].fillna(0.0)
        else:
            df['volume_imbalance'] = 0.0
        
        # Price impact
        if 'volume' in df.columns:
            df['price_impact'] = df['close'].pct_change() / df['volume'].replace(0, np.nan)
            df['price_impact'] = df['price_impact'].fillna(0.0)
        else:
            df['price_impact'] = 0.0
        
        return df
        
    def _calculate_sentiment_features(self, data):
        """Calculate sentiment-based features"""
        df = pd.DataFrame(data)
        # News sentiment
        df['news_sentiment'] = self.sentiment_analyzer.analyze_news('BTCUSDT')
        # Social media sentiment (optioneel, placeholder)
        df['social_sentiment'] = 0.0  # Kan uitgebreid worden met echte analyse
        # Market sentiment (optioneel, placeholder)
        df['market_sentiment'] = 0.0  # Kan uitgebreid worden met echte analyse
        return df
        
    def _calculate_volatility_features(self, data):
        """Calculate volatility-based features"""
        return self.volatility_analyzer.calculate_volatility_features(data)
        
    def _combine_features(self, features):
        """Combine all features into a single feature matrix"""
        combined = pd.concat([
            features['technical'],
            features['microstructure'],
            features['sentiment'],
            features['volatility']
        ], axis=1)
        
        # Handle missing values using ffill() and bfill()
        combined = combined.ffill().bfill().fillna(0)
        
        # Normalize features
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(combined)
        
        return normalized
        
    def train(self, X, y):
        """Train all models in the ensemble"""
        for model in self.models:
            if isinstance(model, Sequential):
                # Neural network training
                model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(patience=5),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
                    ]
                )
            else:
                # Traditional ML model training
                model.fit(X, y)
                
        # Calculate feature importance
        self._update_feature_importance(X, y)
        
    def _calculate_model_weights(self):
        """Calculate optimal weights for each model based on performance"""
        if not self.performance_metrics:
            return [1/len(self.models)] * len(self.models)
            
        # Calculate weights based on model performance
        scores = [metrics['accuracy'] for metrics in self.performance_metrics.values()]
        weights = np.array(scores) / sum(scores)
        return weights
        
    def _update_feature_importance(self, X, y):
        """Update feature importance scores"""
        for model in self.models:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[type(model).__name__] = model.feature_importances_
                
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, np.argmax(predictions, axis=1)),
            'precision': precision_score(y_test, np.argmax(predictions, axis=1), average='weighted'),
            'recall': recall_score(y_test, np.argmax(predictions, axis=1), average='weighted'),
            'f1': f1_score(y_test, np.argmax(predictions, axis=1), average='weighted')
        }
        
        # Update performance metrics
        self.performance_metrics['ensemble'] = metrics
        return metrics
        
    def predict(self, X):
        """Make ensemble predictions with proper input handling"""
        try:
            predictions = []
            weights = self._calculate_model_weights()
            # Prepare sequence data for LSTM if needed
            if len(X.shape) == 2 and self.model_types['lstm'] is not None:
                X_lstm = self._prepare_sequence_data(X)
            else:
                X_lstm = X
            for model, weight in zip(self.models, weights):
                try:
                    if isinstance(model, tf.keras.Model):
                        if self._keras_predict_fn is None:
                            self._keras_predict_fn = self._create_keras_predict_fn(model)
                        # Ensure correct input shape for the model
                        if model == self.model_types['lstm']:
                            pred = self._keras_predict_fn(X_lstm)
                        else:
                            pred = self._keras_predict_fn(X)
                        pred = pred.numpy() if hasattr(pred, 'numpy') else np.array(pred)
                    else:
                        # Traditional ML models
                        pred = model.predict(X_lstm if model == self.model_types['lstm'] else X)
                        if len(pred.shape) == 1:
                            pred = np.eye(3)[pred.astype(int)]
                    predictions.append(pred * weight)
                except Exception as e:
                    self.logger.warning(f"Error in model prediction: {str(e)}")
                    continue
            if not predictions:
                self.logger.error("No valid predictions from any model")
                return np.zeros((X.shape[0], 3))
            target_shape = (X.shape[0], 3)
            aligned_predictions = []
            for pred in predictions:
                if pred.shape != target_shape:
                    if len(pred.shape) == 1:
                        pred = np.eye(3)[pred.astype(int)]
                    elif pred.shape[0] < target_shape[0]:
                        padded = np.zeros(target_shape)
                        padded[:pred.shape[0]] = pred
                        pred = padded
                    elif pred.shape[0] > target_shape[0]:
                        pred = pred[:target_shape[0]]
                aligned_predictions.append(pred)
            ensemble_pred = np.sum(aligned_predictions, axis=0)
            return ensemble_pred
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}")
            return np.zeros((X.shape[0], 3))
        
    def _prepare_sequence_data(self, X):
        """Prepare sequence data for LSTM model"""
        try:
            # Create sliding window sequences
            sequences = []
            for i in range(len(X) - self.sequence_length + 1):
                sequences.append(X[i:i + self.sequence_length])
            
            if not sequences:
                # If not enough data for sequence, pad with zeros
                padded = np.zeros((1, self.sequence_length, X.shape[1]), dtype=np.float32)
                padded[0, -len(X):] = X
                return padded
                
            # Pre-allocate array with correct shape and dtype
            result = np.zeros((len(sequences), self.sequence_length, X.shape[1]), dtype=np.float32)
            for i, seq in enumerate(sequences):
                result[i] = seq
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {str(e)}")
            # Return padded sequence as fallback
            padded = np.zeros((1, self.sequence_length, X.shape[1]), dtype=np.float32)
            padded[0, -len(X):] = X
            return padded
        
    def _calculate_trend_duration(self, trend_series: pd.Series) -> float:
        """Calculate average trend duration"""
        try:
            trend_changes = trend_series.diff().abs()
            durations = []
            current_duration = 0
            
            for change in trend_changes:
                if change == 0:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 1
                    
            return np.mean(durations) if durations else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend duration: {str(e)}")
            return 0.0
            
    def _detect_volatility_regime(self, volatility_series: pd.Series) -> str:
        """Detect current volatility regime"""
        try:
            avg_vol = volatility_series.mean()
            current_vol = volatility_series.iloc[-1]
            
            if current_vol > avg_vol * 1.5:
                return 'high_volatility'
            elif current_vol < avg_vol * 0.5:
                return 'low_volatility'
            else:
                return 'normal_volatility'
                
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {str(e)}")
            return 'unknown'
            
    def _calculate_signal_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate signal accuracy"""
        try:
            # Calculate future returns
            data['future_returns'] = data['close'].shift(-1) / data['close'] - 1
            
            # Calculate signal accuracy
            correct_signals = ((data['signal'] > 0) & (data['future_returns'] > 0)) | \
                            ((data['signal'] < 0) & (data['future_returns'] < 0))
                            
            return correct_signals.mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating signal accuracy: {str(e)}")
            return 0.0

class DataCollector:
    """Handles automatic data collection and preprocessing"""
    def __init__(self, config: dict):
        self.config = config
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.data_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def collect_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Collect historical data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Collect data from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Cache the data
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {str(e)}")
            return pd.DataFrame()
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Zorg dat de index datetime is
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            
            # EMAs
            df['ema_short'] = ta.ema(df['close'], length=9)
            df['ema_medium'] = ta.ema(df['close'], length=21)
            df['ema_long'] = ta.ema(df['close'], length=50)
            
            # VWAP (alleen als we high/low/close/volume hebben)
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                df['vwap'] = ta.vwap(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=df['volume']
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df

    def _get_price_data(self):
        """Get historical price data"""
        try:
            # Check cache first
            cache_file = os.path.join(self.config['cache_dir'], 'price_data.csv')
            if os.path.exists(cache_file):
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if (datetime.now() - data.index[-1]).total_seconds() < 300:  # 5 minutes
                    return data
                    
            # Get data from exchange
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                limit=1000
            )
            
            # Convert to DataFrame
            data = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            # Cache data
            data.to_csv(cache_file)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting price data: {str(e)}")
            return pd.DataFrame()
            
    def _get_volume_data(self):
        """Get historical volume data"""
        try:
            # Get price data (includes volume)
            data = self._get_price_data()
            return data['volume']
            
        except Exception as e:
            logger.error(f"Error getting volume data: {str(e)}")
            return pd.Series()
            
    def _get_current_price(self):
        """Get current price"""
        try:
            # Get latest price data
            data = self._get_price_data()
            return data['close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return 0
            
    def _fetch_news(self, source):
        """Fetch news from specified source"""
        try:
            # Implement news fetching logic for each source
            if source == 'coindesk.com':
                return self._fetch_coindesk_news()
            elif source == 'cointelegraph.com':
                return self._fetch_cointelegraph_news()
            elif source == 'bitcoin.com':
                return self._fetch_bitcoin_news()
            elif source == 'newsbtc.com':
                return self._fetch_newsbtc_news()
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news from {source}: {str(e)}")
            return []
            
    def _fetch_social_posts(self, source):
        """Fetch social media posts from specified source"""
        try:
            # Implement social media fetching logic for each source
            if source == 'twitter':
                return self._fetch_twitter_posts()
            elif source == 'reddit':
                return self._fetch_reddit_posts()
            elif source == 'telegram':
                return self._fetch_telegram_posts()
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error fetching social posts from {source}: {str(e)}")
            return []
            
    def _fetch_options_data(self):
        """Fetch options data"""
        try:
            # Implement options data fetching logic
            # This is a placeholder - implement actual options data fetching
            return None
            
        except Exception as e:
            logger.error(f"Error fetching options data: {str(e)}")
            return None
            
    def _calculate_implied_volatility(self, options_data):
        """Calculate implied volatility from options data"""
        try:
            # Implement implied volatility calculation
            # This is a placeholder - implement actual IV calculation
            return None
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {str(e)}")
            return None

    def _fetch_coindesk_news(self):
        """Fetch news from CoinDesk"""
        try:
            url = 'https://www.coindesk.com/arc/outboundfeeds/rss/'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:  # Get latest 10 articles
                title = item.title.text
                description = item.description.text
                news.append({
                    'title': title,
                    'content': description,
                    'source': 'coindesk.com'
                })
            return news
            
        except Exception as e:
            logger.error(f"Error fetching CoinDesk news: {str(e)}")
            return []
            
    def _fetch_cointelegraph_news(self):
        """Fetch news from Cointelegraph"""
        try:
            url = 'https://cointelegraph.com/rss'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                title = item.title.text
                description = item.description.text
                news.append({
                    'title': title,
                    'content': description,
                    'source': 'cointelegraph.com'
                })
            return news
            
        except Exception as e:
            logger.error(f"Error fetching Cointelegraph news: {str(e)}")
            return []
            
    def _fetch_bitcoin_news(self):
        """Fetch news from Bitcoin.com"""
        try:
            url = 'https://news.bitcoin.com/feed/'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                title = item.title.text
                description = item.description.text
                news.append({
                    'title': title,
                    'content': description,
                    'source': 'bitcoin.com'
                })
            return news
            
        except Exception as e:
            logger.error(f"Error fetching Bitcoin.com news: {str(e)}")
            return []
            
    def _fetch_newsbtc_news(self):
        """Fetch news from NewsBTC"""
        try:
            url = 'https://www.newsbtc.com/feed/'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                title = item.title.text
                description = item.description.text
                news.append({
                    'title': title,
                    'content': description,
                    'source': 'newsbtc.com'
                })
            return news
            
        except Exception as e:
            logger.error(f"Error fetching NewsBTC news: {str(e)}")
            return []
            
    def _fetch_twitter_posts(self):
        """Fetch posts from Twitter"""
        try:
            # This is a placeholder - implement actual Twitter API integration
            # You'll need to set up Twitter API credentials
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Twitter posts: {str(e)}")
            return []
            
    def _fetch_reddit_posts(self):
        """Fetch posts from Reddit"""
        try:
            # This is a placeholder - implement actual Reddit API integration
            # You'll need to set up Reddit API credentials
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {str(e)}")
            return []
            
    def _fetch_telegram_posts(self):
        """Fetch posts from Telegram"""
        try:
            # This is a placeholder - implement actual Telegram API integration
            # You'll need to set up Telegram API credentials
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Telegram posts: {str(e)}")
            return []

class SelfDiagnostic:
    """Handles self-diagnostic and functionality checks"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.health_checks = {}
        self.last_check = None
        
    def run_diagnostics(self) -> Dict:
        """Run comprehensive system diagnostics"""
        try:
            results = {
                'timestamp': datetime.now(),
                'components': {},
                'overall_health': 'healthy',
                'issues': []
            }
            
            # Check data collection
            data_health = self._check_data_collection()
            results['components']['data_collection'] = data_health
            
            # Check model performance
            model_health = self._check_model_performance()
            results['components']['model_performance'] = model_health
            
            # Check trading execution
            execution_health = self._check_trading_execution()
            results['components']['trading_execution'] = execution_health
            
            # Check risk management
            risk_health = self._check_risk_management()
            results['components']['risk_management'] = risk_health
            
            # Update overall health
            if any(comp['status'] == 'critical' for comp in results['components'].values()):
                results['overall_health'] = 'critical'
            elif any(comp['status'] == 'warning' for comp in results['components'].values()):
                results['overall_health'] = 'warning'
                
            self.health_checks = results
            self.last_check = datetime.now()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running diagnostics: {str(e)}")
            return {'overall_health': 'error', 'error': str(e)}
            
    def _check_data_collection(self) -> Dict:
        """Check data collection functionality"""
        try:
            collector = DataCollector(self.config)
            test_symbol = 'BTCUSDT'
            data = collector.collect_historical_data(test_symbol)
            
            return {
                'status': 'healthy' if not data.empty else 'critical',
                'message': 'Data collection working properly' if not data.empty else 'Failed to collect data',
                'details': {
                    'data_points': len(data),
                    'columns': list(data.columns),
                    'last_update': data.index[-1] if not data.empty else None
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Data collection error: {str(e)}',
                'details': {}
            }
            
    def _check_model_performance(self) -> Dict:
        """Check model performance metrics"""
        try:
            # Load performance history
            if os.path.exists('models/performance_history.csv'):
                history = pd.read_csv('models/performance_history.csv')
                
                # Calculate metrics
                recent_performance = history.tail(100)
                win_rate = (recent_performance['pnl'] > 0).mean()
                avg_profit = recent_performance['pnl'].mean()
                sharpe_ratio = recent_performance['pnl'].mean() / recent_performance['pnl'].std()
                
                status = 'healthy'
                if win_rate < 0.4 or avg_profit < 0:
                    status = 'warning'
                if win_rate < 0.3 or avg_profit < -0.02:
                    status = 'critical'
                    
                return {
                    'status': status,
                    'message': 'Model performance within acceptable range' if status == 'healthy' else 'Model performance needs attention',
                    'details': {
                        'win_rate': win_rate,
                        'avg_profit': avg_profit,
                        'sharpe_ratio': sharpe_ratio
                    }
                }
                
            return {
                'status': 'warning',
                'message': 'No performance history available',
                'details': {}
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Error checking model performance: {str(e)}',
                'details': {}
            }
            
    def _check_trading_execution(self) -> Dict:
        """Check trading execution functionality"""
        try:
            # Test order placement
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            
            # Get market data
            ticker = exchange.fetch_ticker('BTCUSDT')
            
            return {
                'status': 'healthy',
                'message': 'Trading execution working properly',
                'details': {
                    'last_price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'volume': ticker['baseVolume']
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Trading execution error: {str(e)}',
                'details': {}
            }
            
    def _check_risk_management(self) -> Dict:
        """Check risk management functionality"""
        try:
            # Load recent trades
            if os.path.exists('models/performance_history.csv'):
                history = pd.read_csv('models/performance_history.csv')
                
                # Calculate risk metrics
                max_drawdown = history['pnl'].cumsum().min()
                volatility = history['pnl'].std()
                avg_position_size = history['position_size'].mean()
                
                status = 'healthy'
                if max_drawdown < -0.1 or volatility > 0.05:
                    status = 'warning'
                if max_drawdown < -0.2 or volatility > 0.1:
                    status = 'critical'
                    
                return {
                    'status': status,
                    'message': 'Risk management within acceptable range' if status == 'healthy' else 'Risk management needs attention',
                    'details': {
                        'max_drawdown': max_drawdown,
                        'volatility': volatility,
                        'avg_position_size': avg_position_size
                    }
                }
                
            return {
                'status': 'warning',
                'message': 'No trade history available for risk analysis',
                'details': {}
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Error checking risk management: {str(e)}',
                'details': {}
            }

class MultiTimeframeAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-timeframe analyzer"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.timeframes = config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.analysis_results = {}
        self.correlation_matrix = pd.DataFrame()
        self.lead_lag_relationships = {}
        self.optimal_combination = {}
        
    def analyze_timeframes(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze relationships between different timeframes"""
        try:
            self.logger.info("Starting multi-timeframe analysis")
            
            # Calculate correlations between timeframes
            self._calculate_correlations(timeframe_data)
            
            # Analyze lead-lag relationships
            self._analyze_lead_lag(timeframe_data)
            
            # Find optimal timeframe combinations
            self._find_optimal_combinations(timeframe_data)
            
            # Calculate timeframe weights
            self._calculate_timeframe_weights()
            
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            return {}
            
    def _calculate_correlations(self, timeframe_data: Dict[str, pd.DataFrame]):
        """Calculate correlation matrix between timeframes"""
        try:
            # Prepare price series for each timeframe
            price_series = {}
            for tf, data in timeframe_data.items():
                if not data.empty:
                    price_series[tf] = data['close'].resample('1h').last().fillna(method='ffill')
            
            # Calculate correlation matrix
            self.correlation_matrix = pd.DataFrame(price_series).corr()
            
            # Store results
            self.analysis_results['correlations'] = self.correlation_matrix.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            
    def _analyze_lead_lag(self, timeframe_data: Dict[str, pd.DataFrame]):
        """Analyze lead-lag relationships between timeframes"""
        try:
            lead_lag = {}
            
            # Analyze each pair of timeframes
            for tf1 in self.timeframes:
                for tf2 in self.timeframes:
                    if tf1 != tf2 and tf1 in timeframe_data and tf2 in timeframe_data:
                        # Calculate cross-correlation
                        data1 = timeframe_data[tf1]['close'].resample('1h').last().fillna(method='ffill')
                        data2 = timeframe_data[tf2]['close'].resample('1h').last().fillna(method='ffill')
                        
                        # Calculate lead-lag relationship
                        max_corr = 0
                        best_lag = 0
                        for lag in range(-24, 25):  # Look for lags up to 24 hours
                            if lag < 0:
                                corr = data1.corr(data2.shift(-lag))
                            else:
                                corr = data1.shift(lag).corr(data2)
                                
                            if abs(corr) > abs(max_corr):
                                max_corr = corr
                                best_lag = lag
                                
                        lead_lag[f"{tf1}_{tf2}"] = {
                            'lag': best_lag,
                            'correlation': max_corr
                        }
            
            self.lead_lag_relationships = lead_lag
            self.analysis_results['lead_lag'] = lead_lag
            
        except Exception as e:
            self.logger.error(f"Error analyzing lead-lag relationships: {str(e)}")
            
    def _find_optimal_combinations(self, timeframe_data: Dict[str, pd.DataFrame]):
        """Find optimal combinations of timeframes"""
        try:
            combinations = []
            
            # Generate all possible combinations of 2-4 timeframes
            for r in range(2, 5):
                for combo in itertools.combinations(self.timeframes, r):
                    # Calculate combined signal
                    signals = []
                    for tf in combo:
                        if tf in timeframe_data and not timeframe_data[tf].empty:
                            # Calculate trend signal
                            data = timeframe_data[tf]
                            ema20 = ta.trend.EMAIndicator(data['close'], 20).ema_indicator()
                            ema50 = ta.trend.EMAIndicator(data['close'], 50).ema_indicator()
                            signal = (ema20 > ema50).astype(int)
                            signals.append(signal)
                    
                    if signals:
                        # Combine signals
                        combined_signal = pd.concat(signals, axis=1).mean(axis=1)
                        
                        # Calculate performance
                        performance = self._calculate_combination_performance(
                            combined_signal, 
                            timeframe_data[combo[0]]['close']
                        )
                        
                        combinations.append({
                            'timeframes': combo,
                            'performance': performance
                        })
            
            # Sort combinations by performance
            combinations.sort(key=lambda x: x['performance'], reverse=True)
            
            # Store results
            self.analysis_results['optimal_combination'] = {
                'best_combination': combinations[0] if combinations else None,
                'all_combinations': combinations
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal combinations: {str(e)}")
            
    def _calculate_combination_performance(self, signal: pd.Series, price: pd.Series) -> float:
        """Calculate performance of a timeframe combination"""
        try:
            # Calculate returns
            returns = price.pct_change()
            
            # Calculate strategy returns
            strategy_returns = returns * signal.shift(1)
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            
            # Combine metrics
            performance = total_return * 0.4 + sharpe_ratio * 0.4 - abs(max_drawdown) * 0.2
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating combination performance: {str(e)}")
            return 0.0
            
    def _calculate_timeframe_weights(self):
        """Calculate optimal weights for each timeframe"""
        try:
            if not self.analysis_results.get('optimal_combination'):
                return
                
            best_combo = self.analysis_results['optimal_combination']['best_combination']
            if not best_combo:
                return
                
            # Calculate weights based on lead-lag relationships and correlations
            weights = {}
            for tf in self.timeframes:
                weight = 0
                count = 0
                
                # Consider lead-lag relationships
                for pair, rel in self.lead_lag_relationships.items():
                    tf1, tf2 = pair.split('_')
                    if tf == tf1:
                        weight += abs(rel['correlation'])
                        count += 1
                    elif tf == tf2:
                        weight += abs(rel['correlation'])
                        count += 1
                
                # Consider correlations
                if tf in self.correlation_matrix.columns:
                    weight += self.correlation_matrix[tf].mean()
                    count += 1
                
                # Calculate final weight
                weights[tf] = weight / count if count > 0 else 0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            self.analysis_results['timeframe_weights'] = weights
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe weights: {str(e)}")

class AdaptiveRiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.initial_balance = config['initial_balance']
        self.max_position_size = config['max_position_size']
        self.max_leverage = config['max_leverage']
        self.risk_free_rate = config['risk_free_rate']
        
        # Initialize risk parameters
        self.risk_parameters = {
            'max_position_size': self.max_position_size,
            'stop_loss': 0.02,  # 2% default stop loss
            'take_profit': 0.04,  # 4% default take profit
            'max_leverage': self.max_leverage,
            'position_sizing_factor': 1.0
        }
        
        # Performance tracking
        self.trade_history = []
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.win_rate = 0
        self.profit_factor = 1.0
        
    def get_risk_parameters(self) -> dict:
        """Get current risk parameters"""
        return self.risk_parameters.copy()
        
    def update_risk_parameters(self, trade_outcome: dict, market_data: dict):
        """Update risk parameters based on trade outcome and market conditions"""
        try:
            # Update trade history
            self.trade_history.append(trade_outcome)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Adjust position size based on performance
            if self.win_rate > 0.6 and self.profit_factor > 1.5:
                self.risk_parameters['position_sizing_factor'] *= 1.1
            elif self.win_rate < 0.4 or self.profit_factor < 1.0:
                self.risk_parameters['position_sizing_factor'] *= 0.9
                
            # Adjust stop loss and take profit based on volatility
            if 'volatility' in market_data:
                vol = market_data['volatility']
                self.risk_parameters['stop_loss'] = max(0.01, min(0.05, vol * 1.5))
                self.risk_parameters['take_profit'] = max(0.02, min(0.1, vol * 3))
                
            # Adjust for market regime
            if 'regime' in market_data:
                regime = market_data['regime']
                if regime == 'trending':
                    self.risk_parameters['position_sizing_factor'] *= 1.2
                elif regime == 'ranging':
                    self.risk_parameters['position_sizing_factor'] *= 0.8
                elif regime == 'volatile':
                    self.risk_parameters['position_sizing_factor'] *= 0.6
                    
            # Ensure parameters stay within bounds
            self.risk_parameters['position_sizing_factor'] = np.clip(
                self.risk_parameters['position_sizing_factor'],
                0.5, 2.0
            )
            
            # Update max position size
            self.risk_parameters['max_position_size'] = (
                self.max_position_size * self.risk_parameters['position_sizing_factor']
            )
            
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {str(e)}")
            
    def _update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trade_history:
            return
            
        # Calculate win rate
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        self.win_rate = wins / len(self.trade_history)
        
        # Calculate profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Update drawdown
        self.current_drawdown = min(0, min(trade['pnl'] for trade in self.trade_history))
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)

class AILearningModule:
    def __init__(self, config):
        """Initialize the AI learning module with enhanced autonomous capabilities"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.feature_data = []
        self.trade_outcomes = []
        self.performance_history = []
        self.last_prediction = None
        self.last_update = None
        self.learning_rate = 0.001
        self.autonomous_mode = True
        self.testing_mode = False
        
        # Initialize components
        self.ensemble = EnsembleModel(config)
        self.risk_manager = AdaptiveRiskManager(config)
        self.mathematical_ai = MathematicalAI(config)
        self.multi_timeframe = MultiTimeframeAnalyzer(config)
        self._initialize_components()
        
        # Initialize autonomous learning parameters
        self.min_data_points = 1000
        self.retraining_interval = pd.Timedelta(hours=24)
        self.optimization_interval = pd.Timedelta(hours=12)
        self.last_optimization = None
        
        # Initialize indicator management
        self.active_indicators = {
            'RSI': {'enabled': True, 'period': 14, 'weight': 1.0},
            'MACD': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9, 'weight': 1.0},
            'Bollinger': {'enabled': True, 'period': 20, 'std_dev': 2, 'weight': 1.0},
            'Volume': {'enabled': True, 'period': 20, 'weight': 1.0},
            'EMA': {'enabled': True, 'periods': [9, 21, 50, 200], 'weight': 1.0},
            'Stochastic': {'enabled': True, 'k_period': 14, 'd_period': 3, 'weight': 1.0},
            'ATR': {'enabled': True, 'period': 14, 'weight': 1.0},
            'Ichimoku': {'enabled': True, 'weight': 1.0}
        }
        
        self.indicator_performance = {}
        self.indicator_optimization_history = []
        
        # Initialize market regime detection
        self.market_regime = 'unknown'
        self.regime_confidence = 0.0
        self.regime_history = []
        
        # Initialize pattern recognition
        self.pattern_recognition = {
            'candlestick_patterns': True,
            'chart_patterns': True,
            'harmonic_patterns': True,
            'elliott_wave': True
        }
        
        # Initialize sentiment analysis
        self.sentiment_analysis = {
            'news_sentiment': True,
            'social_sentiment': True,
            'market_sentiment': True,
            'technical_sentiment': True
        }
        
        # Initialize optimization parameters
        self.optimization_params = {
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }

    def _initialize_components(self):
        """Initialize all AI components with enhanced capabilities"""
        try:
            # Initialize ensemble model with advanced features
            self.ensemble.initialize_models()
            
            # Initialize risk manager with adaptive parameters
            self.risk_manager.initialize()
            
            # Initialize mathematical AI with advanced algorithms
            self.mathematical_ai.initialize()
            
            # Initialize multi-timeframe analyzer
            self.multi_timeframe.initialize()
            
            self.logger.info("AI components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing AI components: {str(e)}")
            raise

    def optimize_strategy(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Optimize trading strategy based on market conditions"""
        try:
            current_time = datetime.now()
            
            # Check if optimization is needed
            if (self.last_optimization is None or 
                current_time - self.last_optimization > self.optimization_interval):
                
                # Detect market regime
                regime = self._detect_market_regime(market_data)
                
                # Optimize indicators for current regime
                indicator_params = self.optimize_indicator_parameters(market_data)
                
                # Optimize timeframe weights
                timeframe_weights = self.multi_timeframe.analyze_timeframes(market_data)
                
                # Optimize risk parameters
                risk_params = self.risk_manager.optimize_parameters(market_data)
                
                # Update strategy components
                self._update_strategy_components(
                    regime=regime,
                    indicator_params=indicator_params,
                    timeframe_weights=timeframe_weights,
                    risk_params=risk_params
                )
                
                self.last_optimization = current_time
                
                return {
                    'regime': regime,
                    'indicator_params': indicator_params,
                    'timeframe_weights': timeframe_weights,
                    'risk_params': risk_params
                }
                
        except Exception as e:
            self.logger.error(f"Error optimizing strategy: {str(e)}")
            return {}

    def _update_strategy_components(self, regime: str, indicator_params: Dict, 
                                  timeframe_weights: Dict, risk_params: Dict):
        """Update strategy components based on optimization results"""
        try:
            # Update market regime
            self.market_regime = regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime
            })
            
            # Update indicator parameters
            for indicator, params in indicator_params.items():
                if indicator in self.active_indicators:
                    self.active_indicators[indicator].update(params)
            
            # Update timeframe weights
            self.multi_timeframe.update_weights(timeframe_weights)
            
            # Update risk parameters
            self.risk_manager.update_parameters(risk_params)
            
            self.logger.info(f"Strategy components updated for regime: {regime}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy components: {str(e)}")

    def predict(self, market_data: dict) -> dict:
        """Generate trading predictions with enhanced autonomous capabilities"""
        try:
            current_time = datetime.now()
            
            # Check if we need to retrain or optimize
            if (self.autonomous_mode and 
                (self.last_retraining is None or 
                 current_time - self.last_retraining > self.retraining_interval)):
                self.train()
                self.last_retraining = current_time
            
            # Optimize strategy if needed
            optimization_results = self.optimize_strategy(market_data)
            
            # Get ensemble predictions
            ensemble_predictions = self.ensemble.predict(market_data)
            
            # Get mathematical AI predictions
            math_signals = self.mathematical_ai.calculate_mathematical_signals(pd.DataFrame(market_data))
            
            # Get multi-timeframe analysis
            timeframe_predictions = {}
            for timeframe, data in market_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Analyze timeframe
                    timeframe_analysis = self.multi_timeframe.analyze_timeframes({timeframe: data})
                    
                    # Generate prediction for this timeframe
                    timeframe_predictions[timeframe] = {
                        'position_size': float(ensemble_predictions['position_size']),
                        'stop_loss': float(ensemble_predictions['stop_loss']),
                        'take_profit': float(ensemble_predictions['take_profit']),
                        'confidence': float(ensemble_predictions['confidence'])
                    }
            
            # Combine predictions from all timeframes
            combined_prediction = self._combine_timeframe_predictions(timeframe_predictions)
            
            # Get risk parameters
            risk_params = self.risk_manager.get_risk_parameters()
            
            # Calculate position size based on combined predictions
            position_size = float(combined_prediction['position_size']) * (1 + combined_prediction['confidence'])
            position_size = np.clip(position_size, -risk_params['max_position_size'], 
                                  risk_params['max_position_size'])
            
            # Calculate stop loss and take profit
            current_price = float(market_data.get('close', 0))
            if current_price > 0:
                stop_loss = current_price * (1 - risk_params['stop_loss'])
                take_profit = current_price * (1 + risk_params['take_profit'])
            else:
                stop_loss = None
                take_profit = None
            
            # Store prediction
            self.last_prediction = {
                'position_size': float(position_size),
                'stop_loss': float(stop_loss) if stop_loss is not None else None,
                'take_profit': float(take_profit) if take_profit is not None else None,
                'confidence': float(combined_prediction['confidence']),
                'market_regime': self.market_regime,
                'optimization_results': optimization_results
            }
            self.last_update = current_time
            
            return self.last_prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return {
                'position_size': 0,
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0,
                'market_regime': 'unknown',
                'optimization_results': {}
            }

    def _detect_market_regime(self, market_data: Dict) -> str:
        """Detect current market regime using multiple indicators"""
        try:
            # Calculate volatility regime
            volatility = self.mathematical_ai.calculate_volatility(market_data)
            volatility_regime = self.mathematical_ai.classify_volatility_regime(volatility)
            
            # Calculate trend regime
            trend = self.mathematical_ai.calculate_trend(market_data)
            trend_regime = self.mathematical_ai.classify_trend_regime(trend)
            
            # Calculate momentum regime
            momentum = self.mathematical_ai.calculate_momentum(market_data)
            momentum_regime = self.mathematical_ai.classify_momentum_regime(momentum)
            
            # Combine regime signals
            regime_signals = {
                'volatility': volatility_regime,
                'trend': trend_regime,
                'momentum': momentum_regime
            }
            
            # Determine dominant regime
            regime_counts = Counter(regime_signals.values())
            dominant_regime = regime_counts.most_common(1)[0][0]
            
            # Calculate regime confidence
            total_signals = len(regime_signals)
            regime_confidence = regime_counts[dominant_regime] / total_signals
            
            self.regime_confidence = regime_confidence
            
            return dominant_regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'

    def _combine_timeframe_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple timeframes with dynamic weighting"""
        try:
            if not predictions:
                return self._get_default_prediction()
            
            # Get timeframe weights from multi-timeframe analyzer
            weights = self.multi_timeframe.analysis_results.get('timeframe_weights', {})
            
            # Initialize combined prediction
            combined = {
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'confidence': 0.0
            }
            
            total_weight = 0.0
            
            # Combine predictions using weighted average
            for timeframe, pred in predictions.items():
                weight = weights.get(timeframe, 1.0)
                total_weight += weight
                
                for key in combined:
                    if pred[key] is not None:
                        combined[key] += pred[key] * weight
            
            # Normalize by total weight
            if total_weight > 0:
                for key in combined:
                    combined[key] /= total_weight
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe predictions: {str(e)}")
            return self._get_default_prediction()

    def update(self, trade_data: Dict, market_data: Dict, outcome: Dict):
        """Update AI model with trade results and market data"""
        try:
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'trade_data': trade_data,
                'market_data': market_data,
                'outcome': outcome
            })
            
            # Update trade outcomes
            self.trade_outcomes.append(outcome)
            
            # Update feature data
            features = self._extract_features(market_data)
            self.feature_data.append(features)
            
            # Adjust learning rate based on performance
            self._adjust_learning_rate()
            
            # Update market regime detection
            self._detect_market_regime(market_data)
            
            # Update indicator performance
            self._update_indicator_performance(trade_data, outcome)
            
            # Update risk parameters
            self.risk_manager.update_risk_parameters(outcome, market_data)
            
            # Retrain if needed
            if (self.autonomous_mode and 
                len(self.performance_history) >= self.min_data_points):
                self.train()
            
        except Exception as e:
            self.logger.error(f"Error updating AI model: {str(e)}")

    def _update_indicator_performance(self, trade_data: Dict, outcome: Dict):
        """Update performance metrics for each indicator"""
        try:
            for indicator, status in self.active_indicators.items():
                if status['enabled']:
                    # Calculate indicator performance
                    performance = self._calculate_indicator_performance(
                        indicator, trade_data, outcome
                    )
                    
                    # Update indicator performance history
                    if indicator not in self.indicator_performance:
                        self.indicator_performance[indicator] = []
                    
                    self.indicator_performance[indicator].append({
                        'timestamp': datetime.now(),
                        'performance': performance
                    })
                    
                    # Update indicator weight based on performance
                    self.active_indicators[indicator]['weight'] = self._calculate_indicator_weight(
                        indicator, performance
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating indicator performance: {str(e)}")

    def _calculate_indicator_performance(self, indicator: str, trade_data: Dict, 
                                      outcome: Dict) -> float:
        """Calculate performance metric for an indicator"""
        try:
            # Get indicator signal
            signal = trade_data.get(f'{indicator}_signal', 0)
            
            # Calculate performance based on trade outcome
            if signal > 0 and outcome['profit'] > 0:
                return 1.0
            elif signal < 0 and outcome['profit'] < 0:
                return 1.0
            elif signal == 0 and outcome['profit'] == 0:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating indicator performance: {str(e)}")
            return 0.0

    def _calculate_indicator_weight(self, indicator: str, performance: float) -> float:
        """Calculate new weight for an indicator based on performance"""
        try:
            # Get current weight
            current_weight = self.active_indicators[indicator]['weight']
            
            # Calculate weight adjustment
            adjustment = (performance - 0.5) * self.learning_rate
            
            # Update weight
            new_weight = current_weight + adjustment
            
            # Ensure weight stays within bounds
            new_weight = np.clip(new_weight, 0.0, 1.0)
            
            return new_weight
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator weight: {str(e)}")
            return self.active_indicators[indicator]['weight']

    def _get_default_prediction(self):
        """Return a default prediction"""
        return {
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence': 0.0
        }
        
    def _adjust_learning_rate(self):
        """Dynamically adjust learning rate based on performance"""
        if len(self.performance_history) < 10:
            return
            
        recent_performance = [p['pnl'] for p in self.performance_history[-10:]]
        avg_performance = np.mean(recent_performance)
        
        if avg_performance > 0.02:  # Good performance
            self.learning_rate *= 0.95  # Slightly decrease
        elif avg_performance < -0.02:  # Poor performance
            self.learning_rate *= 1.05  # Slightly increase
            
        # Keep learning rate within bounds
        self.learning_rate = max(min(self.learning_rate, 0.01), 0.0001)
        
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features from market data"""
        # This method needs to be implemented based on your specific logic
        # For now, we'll return an empty array
        return np.array([])
        
    def _simulate_trades(self, data: pd.DataFrame) -> List[Dict]:
        """Simulate trades on historical data"""
        outcomes = []
        try:
            for i in range(len(data) - 1):
                current_data = data.iloc[i]
                next_data = data.iloc[i + 1]
                
                # Generate prediction
                prediction = self.predict(current_data.to_dict())
                
                # Simulate trade outcome
                if prediction['position_size'] > 0:
                    entry_price = current_data['close']
                    exit_price = next_data['close']
                    pnl = (exit_price - entry_price) / entry_price * prediction['position_size']
                    
                    outcomes.append({
                        'pnl': pnl,
                        'confidence': prediction['confidence'],
                        'position_size': prediction['position_size']
                    })
                    
            return outcomes
            
        except Exception as e:
            self.logger.error(f"Error simulating trades: {str(e)}")
            return []
        
    def train(self):
        """Train all models in the ensemble"""
        for model in self.ensemble.models:
            if isinstance(model, Sequential):
                # Neural network training
                model.fit(
                    self.feature_data, self.trade_outcomes,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(patience=5),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
                    ]
                )
            else:
                # Traditional ML model training
                model.fit(self.feature_data, self.trade_outcomes)
                
        # Calculate feature importance
        self.ensemble._update_feature_importance(self.feature_data, self.trade_outcomes)
        
        # Update performance metrics
        self.ensemble.performance_metrics['ensemble'] = self.ensemble.evaluate(self.feature_data, self.trade_outcomes)
        
        # Update last training time
        self.last_retraining = datetime.now()
        
        self.logger.info("Training completed successfully")
        
    def _calculate_model_weights(self):
        """Calculate optimal weights for each model based on performance"""
        if not self.ensemble.performance_metrics:
            return [1/len(self.ensemble.models)] * len(self.ensemble.models)
            
        # Calculate weights based on model performance
        scores = [metrics['accuracy'] for metrics in self.ensemble.performance_metrics.values()]
        weights = np.array(scores) / sum(scores)
        return weights
        
    def _update_feature_importance(self, X, y):
        """Update feature importance scores"""
        for model in self.ensemble.models:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[type(model).__name__] = model.feature_importances_
                
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        return self.ensemble.evaluate(X_test, y_test)
        
    def _prepare_sequence_data(self, X):
        """Prepare sequence data for LSTM model"""
        try:
            # Create sliding window sequences
            sequences = []
            for i in range(len(X) - self.sequence_length + 1):
                sequences.append(X[i:i + self.sequence_length])
            
            if not sequences:
                # If not enough data for sequence, pad with zeros
                padded = np.zeros((1, self.sequence_length, X.shape[1]), dtype=np.float32)
                padded[0, -len(X):] = X
                return padded
                
            # Pre-allocate array with correct shape and dtype
            result = np.zeros((len(sequences), self.sequence_length, X.shape[1]), dtype=np.float32)
            for i, seq in enumerate(sequences):
                result[i] = seq
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {str(e)}")
            # Return padded sequence as fallback
            padded = np.zeros((1, self.sequence_length, X.shape[1]), dtype=np.float32)
            padded[0, -len(X):] = X
            return padded
        
    def _calculate_trend_duration(self, trend_series: pd.Series) -> float:
        """Calculate average trend duration"""
        try:
            trend_changes = trend_series.diff().abs()
            durations = []
            current_duration = 0
            
            for change in trend_changes:
                if change == 0:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 1
                    
            return np.mean(durations) if durations else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend duration: {str(e)}")
            return 0.0
            
    def _detect_volatility_regime(self, volatility_series: pd.Series) -> str:
        """Detect current volatility regime"""
        try:
            avg_vol = volatility_series.mean()
            current_vol = volatility_series.iloc[-1]
            
            if current_vol > avg_vol * 1.5:
                return 'high_volatility'
            elif current_vol < avg_vol * 0.5:
                return 'low_volatility'
            else:
                return 'normal_volatility'
                
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {str(e)}")
            return 'unknown'
            
    def _calculate_signal_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate signal accuracy"""
        try:
            # Calculate future returns
            data['future_returns'] = data['close'].shift(-1) / data['close'] - 1
            
            # Calculate signal accuracy
            correct_signals = ((data['signal'] > 0) & (data['future_returns'] > 0)) | \
                            ((data['signal'] < 0) & (data['future_returns'] < 0))
                            
            return correct_signals.mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating signal accuracy: {str(e)}")
            return 0.0

class MathematicalAI:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.mathematical_models = {}
        self.initialize_mathematical_models()
        
    def initialize(self):
        self.initialize_mathematical_models()
        
    def initialize_mathematical_models(self):
        try:
            self.mathematical_models = {}
            # Initialize mathematical models based on configuration
            for model_name, model_config in self.config['mathematical_models'].items():
                self.mathematical_models[model_name] = self._create_mathematical_model(model_config)
            self.logger.info("Mathematical AI models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing mathematical models: {str(e)}")
            raise
        
    def _create_mathematical_model(self, model_config):
        # Implement the creation of mathematical models based on configuration
        # This is a placeholder and should be replaced with actual model creation logic
        pass
        
    def calculate_mathematical_signals(self, data):
        signals = {}
        try:
            for model_name, model in self.mathematical_models.items():
                signal = model.predict(data)
                signals[model_name] = signal
            return signals
        except Exception as e:
            self.logger.error(f"Error calculating mathematical signals: {str(e)}")
            return {}
        
    def calculate_volatility(self, data):
        # Implement volatility calculation logic
        pass
        
    def classify_volatility_regime(self, data):
        # Implement volatility regime classification logic
        pass
        
    def calculate_trend(self, data):
        # Implement trend calculation logic
        pass
        
    def classify_trend_regime(self, data):
        # Implement trend regime classification logic
        pass
        
    def calculate_momentum(self, data):
        # Implement momentum calculation logic
        pass
        
    def classify_momentum_regime(self, data):
        # Implement momentum regime classification logic
        pass
        
    def optimize_strategy_weights(self, data):
        # Implement strategy weight optimization logic
        pass
        
    def optimize_risk_parameters(self, data):
        # Implement risk parameter optimization logic
        pass