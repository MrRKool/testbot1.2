import logging
import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class PaperOrder:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime]
    filled_price: Optional[float]
    metadata: Dict

@dataclass
class PaperPosition:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    metadata: Dict

class PaperTrading:
    """Paper trading system for simulating trades without real orders."""
    
    def __init__(self,
                 initial_balance: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Initialize state
        self.balance = initial_balance
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trade_history: List[Dict] = []
        
        # Initialize metrics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def place_order(self,
                   symbol: str,
                   side: OrderSide,
                   order_type: OrderType,
                   quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   metadata: Optional[Dict] = None) -> PaperOrder:
        """Place a paper order."""
        try:
            # Generate order ID
            order_id = f"paper_{len(self.orders) + 1}"
            
            # Create order
            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                filled_at=None,
                filled_price=None,
                metadata=metadata or {}
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Process order
            self._process_order(order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing paper order: {e}")
            raise
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order not found: {order_id}")
                
            order = self.orders[order_id]
            
            if order.status != OrderStatus.PENDING:
                return False
                
            order.status = OrderStatus.CANCELLED
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling paper order: {e}")
            raise
            
    def _process_order(self, order: PaperOrder) -> None:
        """Process a paper order."""
        try:
            # Get current market price
            current_price = self._get_current_price(order.symbol)
            
            # Calculate fill price with slippage
            fill_price = self._calculate_fill_price(
                current_price,
                order.side,
                order.type,
                order.price,
                order.stop_price
            )
            
            # Check if order can be filled
            if not self._can_fill_order(order, fill_price):
                return
                
            # Execute order
            self._execute_order(order, fill_price)
            
        except Exception as e:
            self.logger.error(f"Error processing paper order: {e}")
            raise
            
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        # In a real implementation, this would get the price from market data
        # For now, we'll use a mock price
        return 100.0
        
    def _calculate_fill_price(self,
                            current_price: float,
                            side: OrderSide,
                            order_type: OrderType,
                            limit_price: Optional[float],
                            stop_price: Optional[float]) -> float:
        """Calculate fill price with slippage."""
        try:
            # Determine base price
            if order_type == OrderType.MARKET:
                base_price = current_price
            elif order_type == OrderType.LIMIT:
                base_price = limit_price or current_price
            elif order_type == OrderType.STOP:
                base_price = stop_price or current_price
            elif order_type == OrderType.STOP_LIMIT:
                base_price = limit_price or current_price
            else:
                base_price = current_price
                
            # Apply slippage
            slippage = base_price * self.slippage_rate
            if side == OrderSide.BUY:
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage
                
            return fill_price
            
        except Exception as e:
            self.logger.error(f"Error calculating fill price: {e}")
            raise
            
    def _can_fill_order(self, order: PaperOrder, fill_price: float) -> bool:
        """Check if order can be filled."""
        try:
            # Check balance for buy orders
            if order.side == OrderSide.BUY:
                required_balance = fill_price * order.quantity * (1 + self.commission_rate)
                if required_balance > self.balance:
                    order.status = OrderStatus.REJECTED
                    return False
                    
            # Check position for sell orders
            if order.side == OrderSide.SELL:
                position = self.positions.get(order.symbol)
                if not position or position.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if order can be filled: {e}")
            raise
            
    def _execute_order(self, order: PaperOrder, fill_price: float) -> None:
        """Execute a paper order."""
        try:
            # Calculate commission
            commission = fill_price * order.quantity * self.commission_rate
            self.total_commission += commission
            
            # Update balance
            if order.side == OrderSide.BUY:
                self.balance -= (fill_price * order.quantity + commission)
            else:
                self.balance += (fill_price * order.quantity - commission)
                
            # Update position
            self._update_position(order, fill_price)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_price = fill_price
            
            # Record trade
            self._record_trade(order, fill_price, commission)
            
        except Exception as e:
            self.logger.error(f"Error executing paper order: {e}")
            raise
            
    def _update_position(self, order: PaperOrder, fill_price: float) -> None:
        """Update position after order execution."""
        try:
            position = self.positions.get(order.symbol)
            
            if order.side == OrderSide.BUY:
                if position:
                    # Update existing position
                    total_quantity = position.quantity + order.quantity
                    total_cost = (position.quantity * position.entry_price +
                                order.quantity * fill_price)
                    position.entry_price = total_cost / total_quantity
                    position.quantity = total_quantity
                else:
                    # Create new position
                    self.positions[order.symbol] = PaperPosition(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        entry_price=fill_price,
                        current_price=fill_price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        metadata={}
                    )
            else:
                if position:
                    # Calculate realized PnL
                    realized_pnl = (fill_price - position.entry_price) * order.quantity
                    position.realized_pnl += realized_pnl
                    
                    # Update position
                    position.quantity -= order.quantity
                    if position.quantity == 0:
                        del self.positions[order.symbol]
                        
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise
            
    def _record_trade(self,
                     order: PaperOrder,
                     fill_price: float,
                     commission: float) -> None:
        """Record trade in history."""
        try:
            trade = {
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'fill_price': fill_price,
                'commission': commission,
                'timestamp': order.filled_at,
                'metadata': order.metadata
            }
            
            self.trade_history.append(trade)
            self.total_trades += 1
            
            # Update winning trades count
            if order.side == OrderSide.SELL:
                position = self.positions.get(order.symbol)
                if position and position.realized_pnl > 0:
                    self.winning_trades += 1
                    
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
            raise
            
    def update_positions(self, market_data: Dict[str, float]) -> None:
        """Update positions with current market prices."""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    position.current_price = market_data[symbol]
                    position.unrealized_pnl = (
                        position.current_price - position.entry_price
                    ) * position.quantity
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            raise
            
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for symbol."""
        return self.positions.get(symbol)
        
    def get_positions(self) -> Dict[str, PaperPosition]:
        """Get all positions."""
        return self.positions
        
    def get_orders(self,
                  symbol: Optional[str] = None,
                  status: Optional[OrderStatus] = None) -> List[PaperOrder]:
        """Get orders with optional filtering."""
        try:
            orders = list(self.orders.values())
            
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
                
            if status:
                orders = [o for o in orders if o.status == status]
                
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise
            
    def get_trade_history(self,
                         symbol: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """Get trade history with optional filtering."""
        try:
            trades = self.trade_history
            
            if symbol:
                trades = [t for t in trades if t['symbol'] == symbol]
                
            if start_time:
                trades = [t for t in trades if t['timestamp'] >= start_time]
                
            if end_time:
                trades = [t for t in trades if t['timestamp'] <= end_time]
                
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            raise
            
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        try:
            total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self.positions.values())
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            return {
                'initial_balance': self.initial_balance,
                'current_balance': self.balance,
                'total_pnl': total_pnl,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            raise
            
    def reset(self) -> None:
        """Reset paper trading system."""
        try:
            self.balance = self.initial_balance
            self.positions.clear()
            self.orders.clear()
            self.trade_history.clear()
            self.total_commission = 0.0
            self.total_slippage = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            
        except Exception as e:
            self.logger.error(f"Error resetting paper trading system: {e}")
            raise 