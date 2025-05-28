import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from bot.analysis.market_regime import MarketRegimeDetector
from bot.core.order_manager import Order, OrderManager
from bot.core.bybit_api import BybitAPI

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict]
    equity_curve: pd.Series
    daily_returns: pd.Series

class BacktestEngine:
    """Advanced backtesting engine for trading strategies."""
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.001,    # 0.1%
                 risk_free_rate: float = 0.02):  # 2%
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
        self.market_regime_detector = MarketRegimeDetector()
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        
    def run_backtest(self,
                    strategy: Callable,
                    market_data: Dict[str, pd.DataFrame],
                    start_date: datetime,
                    end_date: datetime) -> BacktestResult:
        """Run backtest for a trading strategy."""
        try:
            # Initialize results
            self.current_capital = self.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = [self.initial_capital]
            
            # Process each symbol
            for symbol, data in market_data.items():
                # Filter data for date range
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data[mask]
                
                # Run strategy for each timeframe
                for i in range(len(data)):
                    current_data = data.iloc[:i+1]
                    if len(current_data) < 2:  # Need at least 2 data points
                        continue
                        
                    # Get current market state
                    current_price = current_data['close'].iloc[-1]
                    current_time = current_data.index[-1]
                    
                    # Detect market regime
                    regime = self.market_regime_detector.detect_regime(current_data)
                    
                    # Get strategy signals
                    signals = strategy(current_data, self.positions, regime)
                    
                    # Process signals
                    if signals:
                        self._process_signals(signals, current_price, current_time)
                        
                    # Update equity curve
                    self._update_equity(current_price)
                    
            # Calculate performance metrics
            metrics = self._calculate_metrics()
            
            return BacktestResult(
                total_trades=metrics['total_trades'],
                winning_trades=metrics['winning_trades'],
                losing_trades=metrics['losing_trades'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_profit=metrics['total_profit'],
                max_drawdown=metrics['max_drawdown'],
                sharpe_ratio=metrics['sharpe_ratio'],
                trades=self.trades,
                equity_curve=pd.Series(self.equity_curve, index=market_data[list(market_data.keys())[0]].index[:len(self.equity_curve)]),
                daily_returns=metrics['daily_returns']
            )
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return None
            
    def _process_signals(self, signals: Dict, current_price: float, current_time: datetime):
        """Process trading signals."""
        try:
            for symbol, signal in signals.items():
                if signal['action'] == 'buy':
                    self._execute_buy(symbol, signal['quantity'], current_price, current_time)
                elif signal['action'] == 'sell':
                    self._execute_sell(symbol, signal['quantity'], current_price, current_time)
                    
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            
    def _execute_buy(self, symbol: str, quantity: float, price: float, time: datetime):
        """Execute buy order."""
        try:
            # Calculate costs
            cost = quantity * price
            commission = cost * self.commission
            slippage_cost = cost * self.slippage
            total_cost = cost + commission + slippage_cost
            
            # Check if we have enough capital
            if total_cost > self.current_capital:
                self.logger.warning(f"Insufficient capital for buy order: {symbol}")
                return
                
            # Update positions and capital
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
                
            self.current_capital -= total_cost
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'cost': total_cost,
                'time': time
            })
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            
    def _execute_sell(self, symbol: str, quantity: float, price: float, time: datetime):
        """Execute sell order."""
        try:
            # Check if we have enough position
            if symbol not in self.positions or self.positions[symbol] < quantity:
                self.logger.warning(f"Insufficient position for sell order: {symbol}")
                return
                
            # Calculate proceeds
            proceeds = quantity * price
            commission = proceeds * self.commission
            slippage_cost = proceeds * self.slippage
            total_proceeds = proceeds - commission - slippage_cost
            
            # Update positions and capital
            self.positions[symbol] -= quantity
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                
            self.current_capital += total_proceeds
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'proceeds': total_proceeds,
                'time': time
            })
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            
    def _update_equity(self, current_price: float):
        """Update equity curve."""
        try:
            # Calculate current position value
            position_value = sum(
                quantity * current_price
                for symbol, quantity in self.positions.items()
            )
            
            # Update equity curve
            self.equity_curve.append(self.current_capital + position_value)
            
        except Exception as e:
            self.logger.error(f"Error updating equity: {e}")
            
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        try:
            # Convert equity curve to returns
            equity_series = pd.Series(self.equity_curve)
            daily_returns = equity_series.pct_change().dropna()
            
            # Calculate basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('proceeds', 0) > t.get('cost', 0)])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit metrics
            total_profit = self.equity_curve[-1] - self.initial_capital
            total_gains = sum(t.get('proceeds', 0) - t.get('cost', 0) for t in self.trades if t.get('proceeds', 0) > t.get('cost', 0))
            total_losses = abs(sum(t.get('proceeds', 0) - t.get('cost', 0) for t in self.trades if t.get('proceeds', 0) <= t.get('cost', 0)))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Calculate drawdown
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Calculate Sharpe ratio
            excess_returns = daily_returns - self.risk_free_rate/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'daily_returns': daily_returns
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {} 