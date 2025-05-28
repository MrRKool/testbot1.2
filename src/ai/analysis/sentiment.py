import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import time
from functools import lru_cache
import threading
from queue import Queue
import json
import os

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()
        
    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            if time_since_last_call < 1.0 / self.calls_per_second:
                time.sleep(1.0 / self.calls_per_second - time_since_last_call)
            self.last_call = time.time()

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources with caching and rate limiting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
        
        # Initialize rate limiters
        self.news_rate_limiter = RateLimiter(calls_per_second=2)
        self.social_rate_limiter = RateLimiter(calls_per_second=5)
        
        # Initialize cache
        self.cache_dir = "cache/sentiment"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_ttl = timedelta(hours=1)
        
        # Initialize queues for async processing
        self.news_queue = Queue()
        self.social_queue = Queue()
        
        # Start worker threads
        self.start_workers()
        
    def start_workers(self):
        """Start worker threads for async processing."""
        self.news_worker = threading.Thread(target=self._process_news_queue, daemon=True)
        self.social_worker = threading.Thread(target=self._process_social_queue, daemon=True)
        self.news_worker.start()
        self.social_worker.start()
        
    def _process_news_queue(self):
        """Process news queue items."""
        while True:
            item = self.news_queue.get()
            if item is None:
                break
            symbol, callback = item
            try:
                sentiment = self._analyze_news_sentiment(symbol)
                callback(sentiment)
            except Exception as e:
                self.logger.error(f"Error processing news for {symbol}: {str(e)}")
            finally:
                self.news_queue.task_done()
                
    def _process_social_queue(self):
        """Process social queue items."""
        while True:
            item = self.social_queue.get()
            if item is None:
                break
            symbol, callback = item
            try:
                sentiment = self._analyze_social_sentiment(symbol)
                callback(sentiment)
            except Exception as e:
                self.logger.error(f"Error processing social for {symbol}: {str(e)}")
            finally:
                self.social_queue.task_done()
                
    @lru_cache(maxsize=100)
    def _get_cached_sentiment(self, symbol: str, source: str) -> Optional[float]:
        """Get cached sentiment if available and not expired."""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{source}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < self.cache_ttl:
                        return data['sentiment']
            except Exception as e:
                self.logger.error(f"Error reading cache for {symbol}: {str(e)}")
        return None
        
    def _save_to_cache(self, symbol: str, source: str, sentiment: float):
        """Save sentiment to cache."""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{source}.json")
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'sentiment': sentiment
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error saving to cache for {symbol}: {str(e)}")
            
    def analyze_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from all sources."""
        try:
            # Check cache first
            cached_sentiment = self._get_cached_sentiment(symbol, 'combined')
            if cached_sentiment is not None:
                return cached_sentiment
                
            # Get sentiments from different sources
            news_sentiment = self._analyze_news_sentiment(symbol)
            social_sentiment = self._analyze_social_sentiment(symbol)
            technical_sentiment = self._analyze_technical_sentiment(symbol)
            
            # Combine sentiments
            combined_sentiment = {
                'news': news_sentiment,
                'social': social_sentiment,
                'technical': technical_sentiment,
                'overall': (news_sentiment + social_sentiment + technical_sentiment) / 3
            }
            
            # Save to cache
            self._save_to_cache(symbol, 'combined', combined_sentiment)
            
            return combined_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {
                'news': 0.0,
                'social': 0.0,
                'technical': 0.0,
                'overall': 0.0
            }
            
    def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from news sources."""
        try:
            # Check cache
            cached = self._get_cached_sentiment(symbol, 'news')
            if cached is not None:
                return cached
                
            # Respect rate limit
            self.news_rate_limiter.wait()
            
            # Get news from different sources
            news_sources = [
                self._fetch_coindesk_news,
                self._fetch_cointelegraph_news,
                self._fetch_bitcoin_news,
                self._fetch_newsbtc_news
            ]
            
            sentiments = []
            for source in news_sources:
                try:
                    news = source()
                    for article in news:
                        text = article['title'] + " " + (article.get('description', ''))
                        result = self.sentiment_analyzer(text)[0]
                        sentiment = 1.0 if result['label'] == 'POSITIVE' else -1.0
                        sentiment *= result['score']
                        sentiments.append(sentiment)
                except Exception as e:
                    self.logger.error(f"Error fetching news: {str(e)}")
                    
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Save to cache
            self._save_to_cache(symbol, 'news', avg_sentiment)
            
            return avg_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return 0.0
            
    def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from social media."""
        try:
            # Check cache
            cached = self._get_cached_sentiment(symbol, 'social')
            if cached is not None:
                return cached
                
            # Respect rate limit
            self.social_rate_limiter.wait()
            
            # Get posts from different sources
            social_sources = [
                self._fetch_twitter_posts,
                self._fetch_reddit_posts,
                self._fetch_telegram_posts
            ]
            
            sentiments = []
            for source in social_sources:
                try:
                    posts = source()
                    for post in posts:
                        text = post['text']
                        result = self.sentiment_analyzer(text)[0]
                        sentiment = 1.0 if result['label'] == 'POSITIVE' else -1.0
                        sentiment *= result['score']
                        sentiments.append(sentiment)
                except Exception as e:
                    self.logger.error(f"Error fetching social posts: {str(e)}")
                    
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Save to cache
            self._save_to_cache(symbol, 'social', avg_sentiment)
            
            return avg_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {str(e)}")
            return 0.0
            
    def _analyze_technical_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from technical indicators."""
        try:
            # Check cache
            cached = self._get_cached_sentiment(symbol, 'technical')
            if cached is not None:
                return cached
                
            # Get price data
            data = self._get_price_data(symbol)
            if data.empty:
                return 0.0
                
            # Calculate technical indicators
            rsi = self._calculate_rsi(data)
            macd = self._calculate_macd(data)
            bb = self._calculate_bollinger_bands(data)
            volume = self._calculate_volume_profile(data)
            
            # Convert to sentiment scores
            rsi_sentiment = self._rsi_to_sentiment(rsi)
            macd_sentiment = self._macd_to_sentiment(macd)
            bb_sentiment = self._bb_to_sentiment(bb)
            volume_sentiment = self._volume_to_sentiment(volume)
            
            # Combine sentiments
            technical_sentiment = np.mean([
                rsi_sentiment,
                macd_sentiment,
                bb_sentiment,
                volume_sentiment
            ])
            
            # Save to cache
            self._save_to_cache(symbol, 'technical', technical_sentiment)
            
            return technical_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical sentiment: {str(e)}")
            return 0.0
            
    def _fetch_coindesk_news(self) -> List[Dict]:
        """Fetch news from CoinDesk."""
        try:
            response = requests.get('https://www.coindesk.com/arc/outboundfeeds/rss/')
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:  # Get latest 10 articles
                news.append({
                    'title': item.title.text,
                    'description': item.description.text,
                    'link': item.link.text,
                    'date': item.pubDate.text
                })
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching CoinDesk news: {str(e)}")
            return []
            
    def _fetch_cointelegraph_news(self) -> List[Dict]:
        """Fetch news from Cointelegraph."""
        try:
            response = requests.get('https://cointelegraph.com/rss')
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                news.append({
                    'title': item.title.text,
                    'description': item.description.text,
                    'link': item.link.text,
                    'date': item.pubDate.text
                })
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching Cointelegraph news: {str(e)}")
            return []
            
    def _fetch_bitcoin_news(self) -> List[Dict]:
        """Fetch news from Bitcoin.com."""
        try:
            response = requests.get('https://news.bitcoin.com/feed/')
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                news.append({
                    'title': item.title.text,
                    'description': item.description.text,
                    'link': item.link.text,
                    'date': item.pubDate.text
                })
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching Bitcoin.com news: {str(e)}")
            return []
            
    def _fetch_newsbtc_news(self) -> List[Dict]:
        """Fetch news from NewsBTC."""
        try:
            response = requests.get('https://www.newsbtc.com/feed/')
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            
            news = []
            for item in items[:10]:
                news.append({
                    'title': item.title.text,
                    'description': item.description.text,
                    'link': item.link.text,
                    'date': item.pubDate.text
                })
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching NewsBTC news: {str(e)}")
            return []
            
    def _fetch_twitter_posts(self) -> List[Dict]:
        """Fetch posts from Twitter."""
        # Note: This is a placeholder. You'll need to implement Twitter API integration
        return []
        
    def _fetch_reddit_posts(self) -> List[Dict]:
        """Fetch posts from Reddit."""
        # Note: This is a placeholder. You'll need to implement Reddit API integration
        return []
        
    def _fetch_telegram_posts(self) -> List[Dict]:
        """Fetch posts from Telegram."""
        # Note: This is a placeholder. You'll need to implement Telegram API integration
        return []
        
    def _get_price_data(self, symbol: str) -> pd.DataFrame:
        """Get historical price data."""
        # Note: This is a placeholder. You'll need to implement price data fetching
        return pd.DataFrame()
        
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return pd.DataFrame({'macd': macd, 'signal': signal})
        
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        middle_band = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return pd.DataFrame({
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        })
        
    def _calculate_volume_profile(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume profile."""
        return data['volume'] / data['volume'].rolling(window=period).mean()
        
    def _rsi_to_sentiment(self, rsi: pd.Series) -> float:
        """Convert RSI to sentiment score."""
        if rsi.empty:
            return 0.0
        last_rsi = rsi.iloc[-1]
        if last_rsi > 70:
            return -1.0
        elif last_rsi < 30:
            return 1.0
        else:
            return 0.0
            
    def _macd_to_sentiment(self, macd_data: pd.DataFrame) -> float:
        """Convert MACD to sentiment score."""
        if macd_data.empty:
            return 0.0
        last_macd = macd_data['macd'].iloc[-1]
        last_signal = macd_data['signal'].iloc[-1]
        if last_macd > last_signal:
            return 1.0
        else:
            return -1.0
            
    def _bb_to_sentiment(self, bb_data: pd.DataFrame) -> float:
        """Convert Bollinger Bands to sentiment score."""
        if bb_data.empty:
            return 0.0
        last_close = bb_data['middle'].iloc[-1]
        last_upper = bb_data['upper'].iloc[-1]
        last_lower = bb_data['lower'].iloc[-1]
        if last_close > last_upper:
            return -1.0
        elif last_close < last_lower:
            return 1.0
        else:
            return 0.0
            
    def _volume_to_sentiment(self, volume_data: pd.Series) -> float:
        """Convert volume profile to sentiment score."""
        if volume_data.empty:
            return 0.0
        last_volume = volume_data.iloc[-1]
        if last_volume > 1.5:
            return 1.0
        elif last_volume < 0.5:
            return -1.0
        else:
            return 0.0 