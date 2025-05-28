import hmac
import hashlib
import time
import json
from typing import Dict, List, Optional, Union
import aiohttp
import logging
from datetime import datetime

class BybitAPI:
    """Bybit API client for handling all API interactions."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        """Setup aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, params: Dict) -> str:
        """Generate signature for API authentication."""
        sorted_params = sorted(params.items())
        param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        signature = hmac.new(
            self.api_secret.encode(),
            param_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                      data: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make API request with proper authentication."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if signed:
            timestamp = int(time.time() * 1000)
            params = params or {}
            params.update({
                "api_key": self.api_key,
                "timestamp": timestamp,
                "recv_window": 5000
            })
            params["sign"] = self._generate_signature(params)
            
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params if method == "GET" else None,
                json=data if method != "GET" else None,
                headers=headers
            ) as response:
                result = await response.json()
                
                if result.get("ret_code") != 0:
                    self.logger.error(f"API Error: {result}")
                    raise Exception(f"API Error: {result.get('ret_msg')}")
                    
                return result.get("result", {})
                
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
            
    async def get_wallet_balance(self) -> Dict:
        """Get wallet balance."""
        return await self._request("GET", "/v5/account/wallet-balance", signed=True)
        
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/position/list", params=params, signed=True)
        
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Dict:
        """Place a new order."""
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(quantity),
            "timeInForce": "GTC"
        }
        
        if price:
            data["price"] = str(price)
        if stop_price:
            data["stopPrice"] = str(stop_price)
            
        return await self._request("POST", "/v5/order/create", data=data, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an existing order."""
        data = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id
        }
        return await self._request("POST", "/v5/order/cancel", data=data, signed=True)
        
    async def get_order_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get order history."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v5/order/realtime", params=params, signed=True)
        
    async def get_klines(self, symbol: str, interval: str, 
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        limit: int = 200) -> List[Dict]:
        """Get kline/candlestick data."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time
            
        return await self._request("GET", "/v5/market/kline", params=params)
        
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information."""
        params = {
            "category": "linear",
            "symbol": symbol
        }
        return await self._request("GET", "/v5/market/tickers", params=params)
        
    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook data."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v5/market/orderbook", params=params)
        
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate."""
        params = {
            "category": "linear",
            "symbol": symbol
        }
        return await self._request("GET", "/v5/market/funding/history", params=params)
        
    async def get_leverage(self, symbol: str) -> Dict:
        """Get current leverage."""
        params = {
            "category": "linear",
            "symbol": symbol
        }
        return await self._request("GET", "/v5/position/leverage", params=params, signed=True)
        
    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol."""
        data = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        return await self._request("POST", "/v5/position/leverage", data=data, signed=True) 