# --------------------------------------------------------
# Deel 1: Imports & API endpoints
# --------------------------------------------------------

import os
import sys
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from functools import wraps, lru_cache
import numpy as np
from utils.rate_limiter import RateLimiter
import hmac
import hashlib
from requests.exceptions import RequestException, Timeout, ConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import asyncio
from cachetools import TTLCache, cached
import diskcache
from typing_extensions import TypedDict

BASE_URL = "https://api.bybit.com/v5/market/tickers"
CANDLES_URL = "https://api.bybit.com/v5/market/kline"

# --------------------------------------------------------
# Logging configuratie
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/price_fetcher.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)
logger.info("📘 Price Fetcher gestart met logging.")

# --------------------------------------------------------
# Types en Constants
# --------------------------------------------------------

class KlineData(TypedDict):
    timestamp: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    turnover: str

class PriceData(TypedDict):
    symbol: str
    price: str
    timestamp: int

# --------------------------------------------------------
# Configuratie
# --------------------------------------------------------

@dataclass
class PriceFetcherConfig:
    """Configuratie voor de price fetcher."""
    # API instellingen
    base_url: str = "https://api.bybit.com/v5"
    timeout: int = 10
    max_retries: int = 3
    retry_delay: int = 1
    
    # Cache instellingen
    cache_enabled: bool = True
    cache_ttl: int = 60  # 1 minuut
    cache_dir: str = "cache"
    cache_size: int = 1000  # Maximum aantal items in cache
    
    # Rate limiting
    calls_per_second: int = 1
    calls_per_minute: int = 30
    calls_per_hour: int = 500
    
    # Data instellingen
    default_timeframes: List[str] = field(default_factory=lambda: ["1", "5", "15", "60", "240", "D"])
    default_limit: int = 200
    
    # Performance
    max_workers: int = 4
    connection_pool_size: int = 10
    use_async: bool = True
    
    # Error handling
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 1.5
    retry_max_delay: int = 10

class PriceFetcher:
    """Klasse voor het ophalen van prijsdata met verbeterde error handling en performance."""
    
    def __init__(self, config: PriceFetcherConfig):
        """Initialiseer de PriceFetcher met configuratie."""
        try:
            self.config = config
            exchange_config = config.get('exchange', {})
            self.base_url = exchange_config.get('base_url', "https://api.bybit.com")
            self.api_key = exchange_config.get('api', {}).get('api_key')
            self.api_secret = exchange_config.get('api', {}).get('api_secret')
            
            # Setup logger
            self.logger = logging.getLogger(__name__)
            
            # Rate limiter configuratie
            rate_limit = exchange_config.get('api', {}).get('rate_limit', {})
            self.rate_limiter = RateLimiter(
                calls_per_second=rate_limit.get('calls_per_second', 1),
                calls_per_minute=rate_limit.get('calls_per_minute', 30),
                calls_per_hour=rate_limit.get('calls_per_hour', 500)
            )
            
            # Cache setup
            self._setup_cache()
            
            # Connection pool
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=config.get('connection_pool_size', 10),
                pool_maxsize=config.get('connection_pool_size', 10),
                max_retries=3
            )
            self.session.mount('https://', adapter)
            
            # Thread pool
            self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
            
            # Async session
            self.async_session = None
            self.loop = None
            
            self.logger.info(f"PriceFetcher geïnitialiseerd met rate limits: {self.rate_limiter.get_current_usage()}")
            
        except Exception as e:
            self.logger.error(f"Kritieke fout bij initialiseren PriceFetcher: {str(e)}")
            raise

    def _setup_cache(self):
        """Setup cache met disk persistence."""
        try:
            cache_dir = Path(self.config.get('cache_dir', 'cache'))
            cache_dir.mkdir(exist_ok=True)
            
            # Memory cache voor snelle toegang
            self.memory_cache = TTLCache(
                maxsize=self.config.get('cache_size', 1000),
                ttl=self.config.get('cache_ttl', 60)
            )
            
            # Disk cache voor persistentie
            self.disk_cache = diskcache.Cache(
                directory=str(cache_dir),
                size_limit=self.config.get('cache_size', 1000) * 1024 * 1024  # 1MB per item
            )
            
            self.logger.info("Cache systemen geïnitialiseerd")
            
        except Exception as e:
            self.logger.error(f"Fout bij setup cache: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, Timeout, ConnectionError))
    )
    async def _make_async_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Maak een asynchrone API request met rate limiting en error handling."""
        try:
            # Wacht indien nodig voor rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Bouw de URL
            url = f"{self.base_url}{endpoint}"
            
            # Voeg API key toe aan headers
            headers = {
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': self._generate_signature(params) if self.api_secret else '',
                'X-BAPI-TIMESTAMP': str(int(time.time() * 1000))
            } if self.api_key else {}
            
            self.logger.debug(f"Making async request to {url}")
            
            try:
                if method.upper() == 'GET':
                    async with self.async_session.get(url, params=params, headers=headers) as response:
                        if response.status != 200:
                            error_msg = f"API request failed: {response.status} - {await response.text()}"
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
                        return await response.json()
                else:
                    async with self.async_session.post(url, json=params, headers=headers) as response:
                        if response.status != 200:
                            error_msg = f"API request failed: {response.status} - {await response.text()}"
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
                        return await response.json()
                            
            except asyncio.TimeoutError:
                self.logger.error("Request timed out after 10 seconds")
                raise
            except aiohttp.ClientError as e:
                self.logger.error(f"Request failed: {str(e)}")
                raise
                    
        except Exception as e:
            self.logger.error(f"Onverwachte error bij async API request: {str(e)}")
            raise

    def _generate_signature(self, params: Dict) -> str:
        """Genereer een signature voor de API request."""
        try:
            if not self.api_secret:
                return ''
                
            # Sorteer parameters
            sorted_params = sorted(params.items())
            query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
            
            # Genereer signature
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            self.logger.error(f"Fout bij genereren signature: {str(e)}")
            raise

    @cached(cache=TTLCache(maxsize=1000, ttl=60))
    async def get_price_async(self, symbol: str) -> float:
        """Asynchrone versie van get_price."""
        try:
            # Check memory cache
            cache_key = f"price_{symbol}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
                
            # Check disk cache
            if cache_key in self.disk_cache:
                price = self.disk_cache[cache_key]
                self.memory_cache[cache_key] = price
                return price
                
            # Haal nieuwe prijs op
            endpoint = '/api/v3/ticker/price'
            params = {'symbol': symbol}
            
            data = await self._make_async_request('GET', endpoint, params)
            price = float(data['price'])
            
            # Update caches
            self.memory_cache[cache_key] = price
            self.disk_cache[cache_key] = price
            
            self.logger.debug(f"Nieuwe prijs opgehaald voor {symbol}: {price}")
            return price
            
        except Exception as e:
            self.logger.error(f"Fout bij ophalen prijs voor {symbol}: {str(e)}")
            raise

    def get_price(self, symbol: str) -> float:
        """Haal de huidige prijs op voor een symbool met caching."""
        if self.config.get('use_async', True):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Als de loop al draait, gebruik create_task
                    return loop.create_task(self.get_price_async(symbol))
                else:
                    # Als de loop niet draait, gebruik run_until_complete
                    return loop.run_until_complete(self.get_price_async(symbol))
            except RuntimeError:
                # Als er geen event loop is, maak een nieuwe
                return asyncio.run(self.get_price_async(symbol))
        else:
            return self._get_price_sync(symbol)

    def _get_price_sync(self, symbol: str) -> float:
        """Synchronische versie van get_price."""
        try:
            # Check memory cache
            cache_key = f"price_{symbol}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
                
            # Check disk cache
            if cache_key in self.disk_cache:
                price = self.disk_cache[cache_key]
                self.memory_cache[cache_key] = price
                return price
                
            # Haal nieuwe prijs op
            endpoint = '/api/v3/ticker/price'
            params = {'symbol': symbol}
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=self.config.get('timeout', 10)
            )
            response.raise_for_status()
            
            data = response.json()
            price = float(data['price'])
            
            # Update caches
            self.memory_cache[cache_key] = price
            self.disk_cache[cache_key] = price
            
            self.logger.debug(f"Nieuwe prijs opgehaald voor {symbol}: {price}")
            return price
            
        except Exception as e:
            self.logger.error(f"Fout bij ophalen prijs voor {symbol}: {str(e)}")
            raise

    async def get_klines_async(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Asynchrone versie van get_klines."""
        try:
            self.logger.info(f"🔄 Ophalen van {limit} {interval} candles voor {symbol}")
            
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol: {symbol}")
            if not interval or not isinstance(interval, str):
                raise ValueError(f"Invalid interval: {interval}")
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"Invalid limit: {limit}")
            
            # Check cache
            cache_key = f"klines_{symbol}_{interval}_{limit}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            if cache_key in self.disk_cache:
                df = self.disk_cache[cache_key]
                self.memory_cache[cache_key] = df
                return df
            
            # Maak API request
            endpoint = "/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            response = await self._make_async_request("GET", endpoint, params)
            
            if not response or 'result' not in response or 'list' not in response['result']:
                self.logger.error(f"Invalid response format: {response}")
                return None
                
            data = response['result']['list']
            if not data:
                self.logger.warning("Empty data list received")
                return None
                
            # Converteer naar DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Converteer numerieke kolommen
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Update caches
            self.memory_cache[cache_key] = df
            self.disk_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Fout bij ophalen klines voor {symbol}: {str(e)}")
            return None

    def get_klines(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Haal klines (candlestick data) op voor een symbol."""
        if self.config.get('use_async', True):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Als de loop al draait, gebruik create_task
                    return loop.create_task(self.get_klines_async(symbol, interval, limit))
                else:
                    # Als de loop niet draait, gebruik run_until_complete
                    return loop.run_until_complete(self.get_klines_async(symbol, interval, limit))
            except RuntimeError:
                # Als er geen event loop is, maak een nieuwe
                return asyncio.run(self.get_klines_async(symbol, interval, limit))
        else:
            return self._get_klines_sync(symbol, interval, limit)

    def _get_klines_sync(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Synchronische versie van get_klines."""
        try:
            self.logger.info(f"🔄 Ophalen van {limit} {interval} candles voor {symbol}")
            
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol: {symbol}")
            if not interval or not isinstance(interval, str):
                raise ValueError(f"Invalid interval: {interval}")
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"Invalid limit: {limit}")
            
            # Check cache
            cache_key = f"klines_{symbol}_{interval}_{limit}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            if cache_key in self.disk_cache:
                df = self.disk_cache[cache_key]
                self.memory_cache[cache_key] = df
                return df
            
            # Maak API request
            endpoint = "/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=self.config.get('timeout', 10)
            )
            response.raise_for_status()
            
            data = response.json()
            if not data or 'result' not in data or 'list' not in data['result']:
                self.logger.error(f"Invalid response format: {data}")
                return None
                
            klines = data['result']['list']
            if not klines:
                self.logger.warning("Empty data list received")
                return None
                
            # Converteer naar DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Converteer numerieke kolommen
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Update caches
            self.memory_cache[cache_key] = df
            self.disk_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Fout bij ophalen klines voor {symbol}: {str(e)}")
            return None

    def clear_cache(self):
        """Clear de prijs cache."""
        try:
            self.memory_cache.clear()
            self.disk_cache.clear()
            self.logger.info("Prijs cache geleegd")
        except Exception as e:
            self.logger.error(f"Fout bij legen cache: {str(e)}")
            raise

    async def close(self):
        """Cleanup async resources."""
        try:
            if self.async_session is not None:
                await self.async_session.close()
                self.async_session = None
                self.logger.info("Async session closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing async session: {str(e)}")

    def __del__(self):
        """Cleanup bij destructie."""
        try:
            self.session.close()
            self.executor.shutdown()
            self.disk_cache.close()
            if self.async_session is not None:
                asyncio.run(self.close())
        except Exception as e:
            self.logger.error(f"Fout bij cleanup: {str(e)}")

    def _initialize_async(self):
        """Initialize async session properly."""
        try:
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                # If loop is running, create a task
                self.loop.create_task(self._setup_async_session())
            else:
                # If loop is not running, run until complete
                self.loop.run_until_complete(self._setup_async_session())
        except Exception as e:
            self.logger.error(f"Error initializing async session: {str(e)}")
            raise

    async def _setup_async_session(self):
        """Setup async session with proper error handling."""
        try:
            if self.async_session is None:
                self.async_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers=self.config.headers if hasattr(self.config, 'headers') else {}
                )
                self.logger.info("Async session initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up async session: {str(e)}")
            raise

    def _setup_connection_pool(self):
        """Setup connection pool for better performance."""
        try:
            # Create a connection pool
            self.conn_pool = aiohttp.TCPConnector(
                limit=100,  # Maximum number of connections
                ttl_dns_cache=300,  # DNS cache TTL in seconds
                use_dns_cache=True,
                force_close=False,
                enable_cleanup_closed=True
            )
            
            # Create a session with the connection pool
            self.async_session = aiohttp.ClientSession(
                connector=self.conn_pool,
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self.config.headers if hasattr(self.config, 'headers') else {}
            )
            
            self.logger.info("Connection pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up connection pool: {str(e)}")
            raise

    async def initialize(self):
        """Initialize the price fetcher asynchronously."""
        try:
            if self.async_session is None:
                # Create a connection pool
                self.conn_pool = aiohttp.TCPConnector(
                    limit=100,  # Maximum number of connections
                    ttl_dns_cache=300,  # DNS cache TTL in seconds
                    use_dns_cache=True,
                    force_close=False,
                    enable_cleanup_closed=True
                )
                
                # Create a session with the connection pool
                self.async_session = aiohttp.ClientSession(
                    connector=self.conn_pool,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers=self.config.headers if hasattr(self.config, 'headers') else {}
                )
                
                self.logger.info("PriceFetcher initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing PriceFetcher: {str(e)}")
            raise

# --------------------------------------------------------
# Main voor testing
# --------------------------------------------------------

async def test_async():
    """Test de price fetcher asynchroon."""
    try:
        # Maak price fetcher instance
        config = {
            'exchange': {
                'base_url': 'https://api.bybit.com',
                'api': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret',
                    'rate_limit': {
                        'calls_per_second': 1,
                        'calls_per_minute': 30,
                        'calls_per_hour': 500
                    }
                }
            }
        }
        fetcher = PriceFetcher(config)
        
        # Test prijs ophalen
        price = await fetcher.get_price_async("BTCUSDT")
        print("BTC prijs:", price)
        
        # Test candles ophalen
        candles = await fetcher.get_klines_async("BTCUSDT", "15")
        print("BTC candles:", candles[:5])
        
        # Cleanup
        await fetcher.close()
        
        print("Price fetcher async test succesvol uitgevoerd")
        
    except Exception as e:
        print(f"Kritieke fout in price fetcher async test: {e}")
        sys.exit(1)

def main():
    """Test de price fetcher."""
    try:
        # Run async tests
        asyncio.run(test_async())
        
        # Maak price fetcher instance voor sync tests
        config = {
            'exchange': {
                'base_url': 'https://api.bybit.com',
                'api': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret',
                    'rate_limit': {
                        'calls_per_second': 1,
                        'calls_per_minute': 30,
                        'calls_per_hour': 500
                    }
                }
            }
        }
        fetcher = PriceFetcher(config)
        
        # Test prijs ophalen
        price = fetcher.get_price("BTCUSDT")
        print("BTC prijs:", price)
        
        # Test candles ophalen
        candles = fetcher.get_klines("BTCUSDT", "15")
        print("BTC candles:", candles[:5])
        
        print("Price fetcher test succesvol uitgevoerd")
        
    except Exception as e:
        print(f"Kritieke fout in price fetcher test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# --------------------------------------------------------
