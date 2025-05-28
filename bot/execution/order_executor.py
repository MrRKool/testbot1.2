import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from enum import Enum
from bot.core.bybit_api import BybitAPI
from bot.analysis.market_regime import MarketRegimeDetector

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "POST_ONLY"

@dataclass
class OrderParams:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    close_on_trigger: bool = False
    iceberg_qty: Optional[float] = None
    iceberg_peak: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    execution_time: Optional[timedelta] = None
    max_slippage: float = 0.001  # 0.1%

@dataclass
class ExecutionResult:
    order_id: str
    status: str
    filled_quantity: float
    average_price: float
    fees: float
    execution_time: datetime
    slippage: float

class OrderExecutor:
    """Advanced order execution system with smart execution strategies."""
    
    def __init__(self,
                 api: BybitAPI,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        
        self.api = api
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        self.market_regime_detector = MarketRegimeDetector()
        self.active_orders: Dict[str, OrderParams] = {}
        
    async def execute_order(self, params: OrderParams) -> ExecutionResult:
        """Execute order with appropriate strategy based on order type."""
        try:
            if params.order_type == OrderType.MARKET:
                return await self._execute_market_order(params)
            elif params.order_type == OrderType.LIMIT:
                return await self._execute_limit_order(params)
            elif params.order_type == OrderType.STOP:
                return await self._execute_stop_order(params)
            elif params.order_type == OrderType.STOP_LIMIT:
                return await self._execute_stop_limit_order(params)
            elif params.order_type == OrderType.TRAILING_STOP:
                return await self._execute_trailing_stop_order(params)
            elif params.order_type == OrderType.ICEBERG:
                return await self._execute_iceberg_order(params)
            elif params.order_type == OrderType.TWAP:
                return await self._execute_twap_order(params)
            elif params.order_type == OrderType.VWAP:
                return await self._execute_vwap_order(params)
            else:
                raise ValueError(f"Unsupported order type: {params.order_type}")
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return None
            
    async def _execute_market_order(self, params: OrderParams) -> ExecutionResult:
        """Execute market order with slippage protection."""
        try:
            # Get current market price
            ticker = await self.api.get_ticker(params.symbol)
            current_price = ticker['last_price']
            
            # Calculate acceptable price range
            if params.side == OrderSide.BUY:
                max_price = current_price * (1 + params.max_slippage)
            else:
                min_price = current_price * (1 - params.max_slippage)
                
            # Place market order
            order = await self.api.place_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.order_type.value,
                qty=params.quantity,
                time_in_force=params.time_in_force.value,
                reduce_only=params.reduce_only,
                close_on_trigger=params.close_on_trigger
            )
            
            # Monitor execution
            execution = await self._monitor_execution(order['order_id'])
            
            # Calculate slippage
            slippage = abs(execution['average_price'] - current_price) / current_price
            
            return ExecutionResult(
                order_id=order['order_id'],
                status=execution['status'],
                filled_quantity=execution['filled_qty'],
                average_price=execution['average_price'],
                fees=execution['fee'],
                execution_time=datetime.now(),
                slippage=slippage
            )
            
        except Exception as e:
            self.logger.error(f"Error executing market order: {e}")
            return None
            
    async def _execute_limit_order(self, params: OrderParams) -> ExecutionResult:
        """Execute limit order with smart price placement."""
        try:
            # Get market data
            market_data = await self.api.get_klines(
                symbol=params.symbol,
                interval='1m',
                limit=100
            )
            
            # Calculate optimal limit price
            optimal_price = self._calculate_optimal_limit_price(
                market_data,
                params.side,
                params.price
            )
            
            # Place limit order
            order = await self.api.place_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.order_type.value,
                qty=params.quantity,
                price=optimal_price,
                time_in_force=params.time_in_force.value,
                reduce_only=params.reduce_only,
                close_on_trigger=params.close_on_trigger
            )
            
            # Monitor execution
            execution = await self._monitor_execution(order['order_id'])
            
            return ExecutionResult(
                order_id=order['order_id'],
                status=execution['status'],
                filled_quantity=execution['filled_qty'],
                average_price=execution['average_price'],
                fees=execution['fee'],
                execution_time=datetime.now(),
                slippage=0.0  # Limit orders have no slippage
            )
            
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}")
            return None
            
    async def _execute_stop_order(self, params: OrderParams) -> ExecutionResult:
        """Execute stop order with dynamic stop price adjustment."""
        try:
            # Get market data
            market_data = await self.api.get_klines(
                symbol=params.symbol,
                interval='1m',
                limit=100
            )
            
            # Calculate optimal stop price
            optimal_stop = self._calculate_optimal_stop_price(
                market_data,
                params.side,
                params.stop_price
            )
            
            # Place stop order
            order = await self.api.place_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.order_type.value,
                qty=params.quantity,
                stop_price=optimal_stop,
                time_in_force=params.time_in_force.value,
                reduce_only=params.reduce_only,
                close_on_trigger=params.close_on_trigger
            )
            
            # Monitor execution
            execution = await self._monitor_execution(order['order_id'])
            
            return ExecutionResult(
                order_id=order['order_id'],
                status=execution['status'],
                filled_quantity=execution['filled_qty'],
                average_price=execution['average_price'],
                fees=execution['fee'],
                execution_time=datetime.now(),
                slippage=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error executing stop order: {e}")
            return None
            
    async def _execute_trailing_stop_order(self, params: OrderParams) -> ExecutionResult:
        """Execute trailing stop order with dynamic trailing distance."""
        try:
            # Get market data
            market_data = await self.api.get_klines(
                symbol=params.symbol,
                interval='1m',
                limit=100
            )
            
            # Calculate optimal trailing distance
            optimal_trail = self._calculate_optimal_trailing_distance(
                market_data,
                params.side,
                params.trailing_stop
            )
            
            # Place trailing stop order
            order = await self.api.place_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.order_type.value,
                qty=params.quantity,
                trailing_stop=optimal_trail,
                time_in_force=params.time_in_force.value,
                reduce_only=params.reduce_only,
                close_on_trigger=params.close_on_trigger
            )
            
            # Monitor execution
            execution = await self._monitor_execution(order['order_id'])
            
            return ExecutionResult(
                order_id=order['order_id'],
                status=execution['status'],
                filled_quantity=execution['filled_qty'],
                average_price=execution['average_price'],
                fees=execution['fee'],
                execution_time=datetime.now(),
                slippage=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error executing trailing stop order: {e}")
            return None
            
    async def _execute_iceberg_order(self, params: OrderParams) -> ExecutionResult:
        """Execute iceberg order with dynamic chunk sizing."""
        try:
            # Calculate optimal chunk size
            optimal_chunk = self._calculate_optimal_iceberg_chunk(
                params.quantity,
                params.iceberg_qty,
                params.iceberg_peak
            )
            
            # Place iceberg order
            order = await self.api.place_order(
                symbol=params.symbol,
                side=params.side.value,
                order_type=params.order_type.value,
                qty=params.quantity,
                price=params.price,
                iceberg_qty=optimal_chunk,
                time_in_force=params.time_in_force.value,
                reduce_only=params.reduce_only,
                close_on_trigger=params.close_on_trigger
            )
            
            # Monitor execution
            execution = await self._monitor_execution(order['order_id'])
            
            return ExecutionResult(
                order_id=order['order_id'],
                status=execution['status'],
                filled_quantity=execution['filled_qty'],
                average_price=execution['average_price'],
                fees=execution['fee'],
                execution_time=datetime.now(),
                slippage=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error executing iceberg order: {e}")
            return None
            
    async def _execute_twap_order(self, params: OrderParams) -> ExecutionResult:
        """Execute TWAP order with dynamic time slicing."""
        try:
            # Calculate optimal time slices
            slices = self._calculate_optimal_twap_slices(
                params.quantity,
                params.execution_time
            )
            
            results = []
            for i, slice_qty in enumerate(slices):
                # Calculate slice execution time
                slice_time = params.execution_time / len(slices)
                
                # Execute slice
                result = await self._execute_market_order(OrderParams(
                    symbol=params.symbol,
                    side=params.side,
                    order_type=OrderType.MARKET,
                    quantity=slice_qty,
                    time_in_force=params.time_in_force,
                    reduce_only=params.reduce_only,
                    close_on_trigger=params.close_on_trigger
                ))
                
                if result:
                    results.append(result)
                    
                # Wait for next slice
                await asyncio.sleep(slice_time.total_seconds())
                
            # Aggregate results
            return self._aggregate_execution_results(results)
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP order: {e}")
            return None
            
    async def _execute_vwap_order(self, params: OrderParams) -> ExecutionResult:
        """Execute VWAP order with volume-based slicing."""
        try:
            # Get historical volume data
            volume_data = await self.api.get_klines(
                symbol=params.symbol,
                interval='1m',
                limit=100
            )
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(volume_data)
            
            # Calculate optimal slices based on volume profile
            slices = self._calculate_optimal_vwap_slices(
                params.quantity,
                volume_profile
            )
            
            results = []
            for slice_qty in slices:
                # Execute slice
                result = await self._execute_market_order(OrderParams(
                    symbol=params.symbol,
                    side=params.side,
                    order_type=OrderType.MARKET,
                    quantity=slice_qty,
                    time_in_force=params.time_in_force,
                    reduce_only=params.reduce_only,
                    close_on_trigger=params.close_on_trigger
                ))
                
                if result:
                    results.append(result)
                    
            # Aggregate results
            return self._aggregate_execution_results(results)
            
        except Exception as e:
            self.logger.error(f"Error executing VWAP order: {e}")
            return None
            
    def _calculate_optimal_limit_price(self,
                                     market_data: pd.DataFrame,
                                     side: OrderSide,
                                     base_price: float) -> float:
        """Calculate optimal limit price based on market conditions."""
        try:
            # Get market regime
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Calculate price levels
            current_price = market_data['close'].iloc[-1]
            volatility = market_data['close'].pct_change().std()
            
            # Adjust price based on regime and side
            if side == OrderSide.BUY:
                if regime['trending'] > 0.5:
                    # In trending market, place limit closer to current price
                    return current_price * (1 - volatility * 0.5)
                else:
                    # In ranging market, place limit further from current price
                    return current_price * (1 - volatility * 2)
            else:
                if regime['trending'] > 0.5:
                    return current_price * (1 + volatility * 0.5)
                else:
                    return current_price * (1 + volatility * 2)
                    
        except Exception as e:
            self.logger.error(f"Error calculating optimal limit price: {e}")
            return base_price
            
    def _calculate_optimal_stop_price(self,
                                    market_data: pd.DataFrame,
                                    side: OrderSide,
                                    base_stop: float) -> float:
        """Calculate optimal stop price based on market conditions."""
        try:
            # Get market regime
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Calculate price levels
            current_price = market_data['close'].iloc[-1]
            volatility = market_data['close'].pct_change().std()
            atr = market_data['high'].sub(market_data['low']).mean()
            
            # Adjust stop based on regime and side
            if side == OrderSide.BUY:
                if regime['volatile'] > 0.5:
                    # In volatile market, use wider stop
                    return current_price * (1 - atr * 2)
                else:
                    # In normal market, use tighter stop
                    return current_price * (1 - atr)
            else:
                if regime['volatile'] > 0.5:
                    return current_price * (1 + atr * 2)
                else:
                    return current_price * (1 + atr)
                    
        except Exception as e:
            self.logger.error(f"Error calculating optimal stop price: {e}")
            return base_stop
            
    def _calculate_optimal_trailing_distance(self,
                                           market_data: pd.DataFrame,
                                           side: OrderSide,
                                           base_trail: float) -> float:
        """Calculate optimal trailing distance based on market conditions."""
        try:
            # Get market regime
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Calculate volatility
            volatility = market_data['close'].pct_change().std()
            atr = market_data['high'].sub(market_data['low']).mean()
            
            # Adjust trailing distance based on regime
            if regime['trending'] > 0.5:
                # In trending market, use wider trail
                return max(base_trail, atr * 2)
            elif regime['volatile'] > 0.5:
                # In volatile market, use even wider trail
                return max(base_trail, atr * 3)
            else:
                # In normal market, use base trail
                return base_trail
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal trailing distance: {e}")
            return base_trail
            
    def _calculate_optimal_iceberg_chunk(self,
                                       total_qty: float,
                                       base_chunk: float,
                                       peak: float) -> float:
        """Calculate optimal iceberg chunk size."""
        try:
            # Calculate average daily volume
            avg_volume = self._get_average_volume()
            
            # Adjust chunk size based on volume
            if avg_volume > 0:
                # Keep chunk size below 5% of average volume
                max_chunk = avg_volume * 0.05
                return min(base_chunk, max_chunk)
            else:
                return base_chunk
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal iceberg chunk: {e}")
            return base_chunk
            
    def _calculate_optimal_twap_slices(self,
                                     total_qty: float,
                                     execution_time: timedelta) -> List[float]:
        """Calculate optimal TWAP slices."""
        try:
            # Calculate number of slices based on execution time
            n_slices = max(1, int(execution_time.total_seconds() / 60))  # One slice per minute
            
            # Calculate base slice size
            base_slice = total_qty / n_slices
            
            # Add some randomness to slice sizes
            slices = []
            remaining_qty = total_qty
            
            for i in range(n_slices - 1):
                # Randomize slice size between 80% and 120% of base slice
                slice_size = base_slice * random.uniform(0.8, 1.2)
                slice_size = min(slice_size, remaining_qty)
                slices.append(slice_size)
                remaining_qty -= slice_size
                
            # Add remaining quantity to last slice
            slices.append(remaining_qty)
            
            return slices
            
        except Exception as e:
            self.logger.error(f"Error calculating TWAP slices: {e}")
            return [total_qty]
            
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile for VWAP execution."""
        try:
            # Calculate average volume per time period
            volume_profile = market_data.groupby(
                market_data.index.hour
            )['volume'].mean()
            
            # Normalize volume profile
            total_volume = volume_profile.sum()
            if total_volume > 0:
                volume_profile = volume_profile / total_volume
                
            return volume_profile.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}
            
    def _calculate_optimal_vwap_slices(self,
                                     total_qty: float,
                                     volume_profile: Dict[str, float]) -> List[float]:
        """Calculate optimal VWAP slices based on volume profile."""
        try:
            if not volume_profile:
                return [total_qty]
                
            # Calculate slices based on volume profile
            slices = []
            remaining_qty = total_qty
            
            for hour, volume_ratio in volume_profile.items():
                slice_qty = total_qty * volume_ratio
                slice_qty = min(slice_qty, remaining_qty)
                slices.append(slice_qty)
                remaining_qty -= slice_qty
                
            return slices
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP slices: {e}")
            return [total_qty]
            
    async def _monitor_execution(self, order_id: str) -> Dict:
        """Monitor order execution status."""
        try:
            for _ in range(self.max_retries):
                order = await self.api.get_order(order_id)
                
                if order['status'] in ['Filled', 'Cancelled', 'Rejected']:
                    return order
                    
                await asyncio.sleep(self.retry_delay)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error monitoring execution: {e}")
            return None
            
    def _aggregate_execution_results(self,
                                   results: List[ExecutionResult]) -> ExecutionResult:
        """Aggregate multiple execution results into one."""
        try:
            if not results:
                return None
                
            total_qty = sum(r.filled_quantity for r in results)
            total_value = sum(r.filled_quantity * r.average_price for r in results)
            total_fees = sum(r.fees for r in results)
            
            return ExecutionResult(
                order_id=results[0].order_id,
                status='Filled',
                filled_quantity=total_qty,
                average_price=total_value / total_qty if total_qty > 0 else 0,
                fees=total_fees,
                execution_time=datetime.now(),
                slippage=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating execution results: {e}")
            return None
            
    def _get_average_volume(self) -> float:
        """Get average daily volume for the symbol."""
        try:
            # This should be implemented based on your data source
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting average volume: {e}")
            return 0.0 