import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
import websockets
from datetime import datetime

class WebSocketManager:
    """Manages WebSocket connections to Bybit for real-time market data."""
    
    def __init__(self, base_url: str = "wss://stream.bybit.com/v5/public/linear"):
        self.base_url = base_url
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        self.logger = logging.getLogger(__name__)
        self.callbacks: Dict[str, List[Callable]] = {}
        
    async def connect(self, symbol: str) -> bool:
        """Establish WebSocket connection for a symbol."""
        try:
            if symbol in self.connections and not self.connections[symbol].closed:
                return True
                
            self.connections[symbol] = await websockets.connect(self.base_url)
            self.reconnect_attempts[symbol] = 0
            self.logger.info(f"WebSocket connection established for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket for {symbol}: {e}")
            return False
            
    async def subscribe(self, symbol: str, channels: List[str]) -> bool:
        """Subscribe to specific channels for a symbol."""
        try:
            if symbol not in self.connections:
                if not await self.connect(symbol):
                    return False
                    
            subscription_msg = {
                "op": "subscribe",
                "args": [f"{channel}.{symbol}" for channel in channels]
            }
            
            await self.connections[symbol].send(json.dumps(subscription_msg))
            self.subscriptions[symbol] = channels
            self.logger.info(f"Subscribed to {channels} for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {channels} for {symbol}: {e}")
            return False
            
    async def handle_message(self, symbol: str, message: str):
        """Process incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if "topic" in data:
                topic = data["topic"]
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        await callback(data)
                        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message for {symbol}: {e}")
            
    async def add_callback(self, topic: str, callback: Callable):
        """Add callback function for specific topic."""
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        self.callbacks[topic].append(callback)
        
    async def start_listening(self, symbol: str):
        """Start listening for messages on WebSocket connection."""
        while True:
            try:
                if symbol not in self.connections or self.connections[symbol].closed:
                    if not await self.connect(symbol):
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                        
                message = await self.connections[symbol].recv()
                await self.handle_message(symbol, message)
                
            except websockets.ConnectionClosed:
                self.logger.warning(f"WebSocket connection closed for {symbol}")
                if await self._handle_reconnect(symbol):
                    continue
                break
            except Exception as e:
                self.logger.error(f"Error in WebSocket listener for {symbol}: {e}")
                if await self._handle_reconnect(symbol):
                    continue
                break
                
    async def _handle_reconnect(self, symbol: str) -> bool:
        """Handle reconnection logic."""
        if symbol not in self.reconnect_attempts:
            self.reconnect_attempts[symbol] = 0
            
        if self.reconnect_attempts[symbol] < self.max_reconnect_attempts:
            self.reconnect_attempts[symbol] += 1
            self.logger.info(f"Attempting to reconnect {symbol} (attempt {self.reconnect_attempts[symbol]})")
            await asyncio.sleep(self.reconnect_delay)
            return await self.connect(symbol)
        else:
            self.logger.error(f"Max reconnection attempts reached for {symbol}")
            return False
            
    async def close(self, symbol: str):
        """Close WebSocket connection for a symbol."""
        try:
            if symbol in self.connections:
                await self.connections[symbol].close()
                del self.connections[symbol]
                del self.subscriptions[symbol]
                self.logger.info(f"Closed WebSocket connection for {symbol}")
        except Exception as e:
            self.logger.error(f"Error closing WebSocket for {symbol}: {e}")
            
    async def close_all(self):
        """Close all WebSocket connections."""
        for symbol in list(self.connections.keys()):
            await self.close(symbol) 