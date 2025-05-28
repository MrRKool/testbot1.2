import aiohttp
import hmac
import hashlib
import time
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
import asyncio
from urllib.parse import urlencode

class Exchange:
    def __init__(self, config: dict):
        """Initialize the exchange interface."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.api_config = config['exchange']['api']
        self.session = None
        self.rate_limits = {
            'calls_per_second': 0,
            'calls_per_minute': 0,
            'calls_per_hour': 0,
            'last_call': datetime.min
        }
        
        self.logger.info("Exchange interface initialized")

    async def initialize(self):
        """Initialize async session."""
        try:
            self.session = aiohttp.ClientSession()
            self.logger.info("Exchange session started")
        except Exception as e:
            self.logger.error(f"Error initializing exchange session: {str(e)}")
            raise

    async def close(self):
        """Close async session."""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("Exchange session closed")
        except Exception as e:
            self.logger.error(f"Error closing exchange session: {str(e)}")

    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        try:
            now = datetime.now()
            elapsed = (now - self.rate_limits['last_call']).total_seconds()
            
            # Reset counters if enough time has passed
            if elapsed >= 3600:
                self.rate_limits['calls_per_hour'] = 0
            if elapsed >= 60:
                self.rate_limits['calls_per_minute'] = 0
            if elapsed >= 1:
                self.rate_limits['calls_per_second'] = 0
            
            # Check limits
            if self.rate_limits['calls_per_second'] >= self.api_config['rate_limit']['calls_per_second']:
                await asyncio.sleep(1)
            if self.rate_limits['calls_per_minute'] >= self.api_config['rate_limit']['calls_per_minute']:
                await asyncio.sleep(60)
            if self.rate_limits['calls_per_hour'] >= self.api_config['rate_limit']['calls_per_hour']:
                await asyncio.sleep(3600)
            
            # Update counters
            self.rate_limits['calls_per_second'] += 1
            self.rate_limits['calls_per_minute'] += 1
            self.rate_limits['calls_per_hour'] += 1
            self.rate_limits['last_call'] = now
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            raise

    def _generate_signature(self, params: Dict) -> str:
        """Generate API signature."""
        try:
            # Sort parameters
            sorted_params = sorted(params.items())
            query_string = urlencode(sorted_params)
            
            # Generate signature
            signature = hmac.new(
                self.api_config['api_secret'].encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Error generating signature: {str(e)}")
            raise

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make API request with rate limiting and retries."""
        try:
            # Prepare request
            url = f"{self.api_config['base_url']}{endpoint}"
            headers = {'Content-Type': 'application/json'}
            
            if signed:
                # Add timestamp and API key
                params = params or {}
                params['timestamp'] = int(time.time() * 1000)
                params['api_key'] = self.api_config['api_key']
                
                # Add signature
                params['sign'] = self._generate_signature(params)
            
            # Make request with retries
            for attempt in range(self.api_config['max_retries']):
                try:
                    await self._check_rate_limit()
                    
                    async with self.session.request(method, url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['ret_code'] == 0:
                                return data['result']
                            else:
                                raise Exception(f"API error: {data['ret_msg']}")
                        else:
                            raise Exception(f"HTTP error: {response.status}")
                            
                except Exception as e:
                    if attempt == self.api_config['max_retries'] - 1:
                        raise
                    await asyncio.sleep(self.api_config['retry_delay'])
                    
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}")
            raise

    async def get_account_balance(self) -> Dict:
        """Get account balance."""
        try:
            return await self._make_request('GET', '/account/balance', signed=True)
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            raise

    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for symbol."""
        try:
            params = {'symbol': symbol}
            return await self._make_request('GET', '/market/ticker', params)
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook for symbol."""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            return await self._make_request('GET', '/market/orderbook', params)
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {str(e)}")
            raise

    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str, price: float = None, stop_price: float = None) -> Dict:
        """Place an order."""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'qty': str(quantity),
                'type': order_type.upper()
            }
            
            if price:
                params['price'] = str(price)
            if stop_price:
                params['stop_price'] = str(stop_price)
            
            return await self._make_request('POST', '/order/create', params, signed=True)
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order."""
        try:
            params = {
                'symbol': symbol,
                'order_id': order_id
            }
            return await self._make_request('POST', '/order/cancel', params, signed=True)
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            raise

    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status."""
        try:
            params = {
                'symbol': symbol,
                'order_id': order_id
            }
            return await self._make_request('GET', '/order/status', params, signed=True)
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            raise

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders."""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            return await self._make_request('GET', '/order/open', params, signed=True)
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")
            raise

    async def get_position(self, symbol: str) -> Dict:
        """Get position for symbol."""
        try:
            params = {'symbol': symbol}
            return await self._make_request('GET', '/position/list', params, signed=True)
        except Exception as e:
            self.logger.error(f"Error getting position: {str(e)}")
            raise

    async def get_klines(self, symbol: str, interval: str, limit: int = 1000, start_time: int = None, end_time: int = None) -> List[Dict]:
        """Get klines/candlestick data."""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            if start_time:
                params['from'] = start_time
            if end_time:
                params['to'] = end_time
                
            return await self._make_request('GET', '/market/kline', params)
            
        except Exception as e:
            self.logger.error(f"Error getting klines: {str(e)}")
            raise 