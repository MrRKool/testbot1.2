import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from .bybit_api import BybitAPI

@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good Till Cancel
    reduce_only: bool = False
    close_on_trigger: bool = False
    order_id: Optional[str] = None
    status: str = 'new'
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class RateLimiter:
    """Implements rate limiting for API calls."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire rate limit token."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests
            self.requests = [req for req in self.requests if now - req < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            self.requests.append(now)
            
class ExecutionEngine:
    """Handles order execution and management."""
    
    def __init__(self, api: BybitAPI):
        self.api = api
        self.active_orders: Dict[str, Order] = {}
        self.logger = logging.getLogger(__name__)
        
    async def execute_order(self, order: Order) -> bool:
        """Execute an order."""
        try:
            # Execute order with Bybit API
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price
            )
            
            if result and 'orderId' in result:
                order.order_id = result['orderId']
                order.status = 'submitted'
                self.active_orders[order.order_id] = order
                self.logger.info(f"Executed order {order.order_id} for {order.symbol}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to execute order: {e}")
            return False
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                result = await self.api.cancel_order(
                    symbol=order.symbol,
                    order_id=order_id
                )
                
                if result and result.get('status') == 'success':
                    del self.active_orders[order_id]
                    self.logger.info(f"Cancelled order {order_id}")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def update_order_status(self, order_id: str, status: str):
        """Update order status."""
        if order_id in self.active_orders:
            self.active_orders[order_id].status = status
            self.active_orders[order_id].updated_at = datetime.now()
            
class OrderManager:
    """Manages order execution with rate limiting."""
    
    def __init__(self, api: BybitAPI, max_requests: int = 100, time_window: int = 60):
        self.api = api
        self.rate_limiter = RateLimiter(max_requests, time_window)
        self.execution_engine = ExecutionEngine(api)
        self.order_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.retry_delay = 1
        
    async def place_order(self, order: Order) -> bool:
        """Place an order with rate limiting."""
        try:
            await self.rate_limiter.acquire()
            return await self.execution_engine.execute_order(order)
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return False
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with rate limiting."""
        try:
            await self.rate_limiter.acquire()
            return await self.execution_engine.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
            
    async def manage_order_queue(self):
        """Process orders in the queue."""
        while True:
            try:
                order = await self.order_queue.get()
                retries = 0
                success = False
                
                while not success and retries < self.max_retries:
                    success = await self.place_order(order)
                    if not success:
                        retries += 1
                        self.logger.warning(f"Failed to process order {order.order_id}, retry {retries}/{self.max_retries}")
                        await asyncio.sleep(self.retry_delay)
                
                if not success:
                    self.logger.error(f"Failed to process order {order.order_id} after {self.max_retries} retries")
                    
                await asyncio.sleep(0.1)  # Prevent CPU overuse
                
            except Exception as e:
                self.logger.error(f"Error in order queue processing: {e}")
                await asyncio.sleep(1)  # Wait before retrying
                
    async def add_to_queue(self, order: Order):
        """Add an order to the processing queue."""
        await self.order_queue.put(order)
        
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders."""
        return self.execution_engine.active_orders.copy()
        
    async def update_order_statuses(self):
        """Update status of all active orders."""
        try:
            await self.rate_limiter.acquire()
            for order_id, order in self.execution_engine.active_orders.items():
                try:
                    result = await self.api.get_order_history(
                        symbol=order.symbol,
                        limit=1
                    )
                    
                    if result and len(result) > 0:
                        latest_order = result[0]
                        if latest_order['orderId'] == order_id:
                            await self.execution_engine.update_order_status(
                                order_id,
                                latest_order['status']
                            )
                            
                except Exception as e:
                    self.logger.error(f"Failed to update status for order {order_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update order statuses: {e}") 