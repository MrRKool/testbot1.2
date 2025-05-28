import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from bot.analysis.market_regime import MarketRegimeDetector
from bot.core.order_manager import Order, OrderManager
from bot.core.bybit_api import BybitAPI
import json
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
from collections import deque
import time
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    type: TradeType
    size: float
    pnl: Optional[float]
    status: str
    metadata: Dict

class BacktestEngine:
    """Advanced backtesting engine for trading strategies."""
    
    def __init__(self, config: Dict):
        """Initialize backtesting engine with configuration."""
        self.log = logging.getLogger(__name__)
        self.config = config
        
        # Initialize core components
        self._init_components()
        
        # Initialize performance tracking
        self._init_performance_tracking()
        
        # Initialize results storage
        self._init_results_storage()
        
    def _init_components(self):
        """Initialize core components."""
        # Trading parameters
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.position_size = self.config.get('position_size', 0.1)  # 10% of capital
        self.max_positions = self.config.get('max_positions', 5)
        self.commission = self.config.get('commission', 0.001)  # 0.1%
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05%
        
        # Risk management
        self.stop_loss = self.config.get('stop_loss', 0.02)  # 2%
        self.take_profit = self.config.get('take_profit', 0.04)  # 4%
        self.max_drawdown = self.config.get('max_drawdown', 0.2)  # 20%
        
        # Performance tracking
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Thread safety
        self._lock = threading.RLock()
        
    def _init_performance_tracking(self):
        """Initialize performance tracking metrics."""
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0
        }
        
    def _init_results_storage(self):
        """Initialize results storage."""
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'trades').mkdir(exist_ok=True)
        (self.results_dir / 'equity').mkdir(exist_ok=True)
        (self.results_dir / 'metrics').mkdir(exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        
    def run_backtest(self, data: pd.DataFrame, strategy: Any) -> Dict:
        """Run backtest with given data and strategy."""
        try:
            self.log.info("Starting backtest...")
            
            # Reset state
            self._reset_state()
            
            # Prepare data
            data = self._prepare_data(data)
            
            # Run simulation
            self._run_simulation(data, strategy)
            
            # Calculate metrics
            self._calculate_metrics()
            
            # Save results
            self._save_results()
            
            # Generate plots
            self._generate_plots()
            
            self.log.info("Backtest completed successfully")
            return {
                'metrics': self.metrics,
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'drawdown_curve': self.drawdown_curve
            }
            
        except Exception as e:
            self.log.error(f"Error running backtest: {str(e)}")
            raise
            
    def _reset_state(self):
        """Reset engine state."""
        self.trades = []
        self.positions = {}
        self.equity_curve = [self.initial_capital]
        self.drawdown_curve = [0.0]
        self._init_performance_tracking()
        
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        try:
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            # Sort by time
            data = data.sort_index()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Handle missing values
            data = data.fillna(method='ffill')
            
            return data
            
        except Exception as e:
            self.log.error(f"Error preparing data: {str(e)}")
            raise
            
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data."""
        try:
            # Moving averages
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['SMA_50'] = data['close'].rolling(window=50).mean()
            data['SMA_200'] = data['close'].rolling(window=200).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['BB_middle'] = data['close'].rolling(window=20).mean()
            data['BB_std'] = data['close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
            data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
            
            return data
            
        except Exception as e:
            self.log.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def _run_simulation(self, data: pd.DataFrame, strategy: Any):
        """Run trading simulation."""
        try:
            current_capital = self.initial_capital
            
            for timestamp, row in tqdm(data.iterrows(), total=len(data), desc="Running simulation"):
                # Update positions
                self._update_positions(row, timestamp)
                
                # Get strategy signals
                signals = strategy.generate_signals(row)
                
                # Execute trades
                if signals:
                    self._execute_trades(signals, row, timestamp, current_capital)
                    
                # Update equity curve
                current_capital = self._update_equity(row, timestamp, current_capital)
                
        except Exception as e:
            self.log.error(f"Error running simulation: {str(e)}")
            raise
            
    def _update_positions(self, data: pd.Series, timestamp: datetime):
        """Update existing positions."""
        try:
            for position_id, position in list(self.positions.items()):
                # Check stop loss
                if self._check_stop_loss(position, data):
                    self._close_position(position_id, data, timestamp, "stop_loss")
                    
                # Check take profit
                elif self._check_take_profit(position, data):
                    self._close_position(position_id, data, timestamp, "take_profit")
                    
        except Exception as e:
            self.log.error(f"Error updating positions: {str(e)}")
            
    def _check_stop_loss(self, position: Trade, data: pd.Series) -> bool:
        """Check if position hit stop loss."""
        try:
            if position.type == TradeType.LONG:
                return data['low'] <= position.entry_price * (1 - self.stop_loss)
            else:
                return data['high'] >= position.entry_price * (1 + self.stop_loss)
        except Exception as e:
            self.log.error(f"Error checking stop loss: {str(e)}")
            return False
            
    def _check_take_profit(self, position: Trade, data: pd.Series) -> bool:
        """Check if position hit take profit."""
        try:
            if position.type == TradeType.LONG:
                return data['high'] >= position.entry_price * (1 + self.take_profit)
            else:
                return data['low'] <= position.entry_price * (1 - self.take_profit)
        except Exception as e:
            self.log.error(f"Error checking take profit: {str(e)}")
            return False
            
    def _execute_trades(self, signals: Dict, data: pd.Series, timestamp: datetime, current_capital: float):
        """Execute trades based on signals."""
        try:
            for signal in signals:
                # Check if we can open new position
                if len(self.positions) >= self.max_positions:
                    continue
                    
                # Calculate position size
                position_size = current_capital * self.position_size
                
                # Create trade
                trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=data['close'],
                    exit_price=None,
                    type=signal.type,
                    size=position_size,
                    pnl=None,
                    status='open',
                    metadata=getattr(signal, 'metadata', {})
                )
                
                # Add to positions
                position_id = len(self.positions)
                self.positions[position_id] = trade
                
        except Exception as e:
            self.log.error(f"Error executing trades: {str(e)}")
            
    def _close_position(self, position_id: int, data: pd.Series, timestamp: datetime, reason: str):
        """Close a position."""
        try:
            position = self.positions[position_id]
            
            # Calculate exit price with slippage
            if position.type == TradeType.LONG:
                exit_price = data['close'] * (1 - self.slippage)
            else:
                exit_price = data['close'] * (1 + self.slippage)
                
            # Calculate PnL
            if position.type == TradeType.LONG:
                pnl = (exit_price - position.entry_price) / position.entry_price
            else:
                pnl = (position.entry_price - exit_price) / position.entry_price
                
            # Update trade
            position.exit_time = timestamp
            position.exit_price = exit_price
            position.pnl = pnl
            position.status = 'closed'
            position.metadata['exit_reason'] = reason
            
            # Move to trades list
            self.trades.append(position)
            del self.positions[position_id]
            
        except Exception as e:
            self.log.error(f"Error closing position: {str(e)}")
            
    def _update_equity(self, data: pd.Series, timestamp: datetime, current_capital: float) -> float:
        """Update equity curve."""
        try:
            # Calculate position values
            position_value = sum(
                position.size * (1 + position.pnl if position.pnl else 0)
                for position in self.positions.values()
            )
            
            # Update equity
            equity = current_capital + position_value
            self.equity_curve.append(equity)
            
            # Update drawdown
            peak = max(self.equity_curve)
            drawdown = (peak - equity) / peak
            self.drawdown_curve.append(drawdown)
            
            return equity
            
        except Exception as e:
            self.log.error(f"Error updating equity: {str(e)}")
            return current_capital
            
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        try:
            if not self.trades:
                return
                
            # Basic metrics
            self.metrics['total_trades'] = len(self.trades)
            self.metrics['winning_trades'] = len([t for t in self.trades if t.pnl > 0])
            self.metrics['losing_trades'] = len([t for t in self.trades if t.pnl < 0])
            
            # Win rate
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
            
            # Profit metrics
            winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
            losing_trades = [t.pnl for t in self.trades if t.pnl < 0]
            
            self.metrics['average_win'] = np.mean(winning_trades) if winning_trades else 0
            self.metrics['average_loss'] = np.mean(losing_trades) if losing_trades else 0
            self.metrics['largest_win'] = max(winning_trades) if winning_trades else 0
            self.metrics['largest_loss'] = min(losing_trades) if losing_trades else 0
            
            # Profit factor
            total_profit = sum(winning_trades)
            total_loss = abs(sum(losing_trades))
            self.metrics['profit_factor'] = total_profit / total_loss if total_loss else float('inf')
            
            # Return metrics
            self.metrics['total_return'] = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
            
            # Annualized return
            days = (self.trades[-1].exit_time - self.trades[0].entry_time).days
            self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (365 / days) - 1
            
            # Risk metrics
            self.metrics['max_drawdown'] = max(self.drawdown_curve)
            
            # Calculate Sharpe ratio
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            self.metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
            
            # Calculate Sortino ratio
            downside_returns = returns[returns < 0]
            self.metrics['sortino_ratio'] = np.sqrt(252) * returns.mean() / downside_returns.std()
            
        except Exception as e:
            self.log.error(f"Error calculating metrics: {str(e)}")
            
    def _save_results(self):
        """Save backtest results."""
        try:
            # Save trades
            trades_df = pd.DataFrame([vars(trade) for trade in self.trades])
            trades_df.to_csv(self.results_dir / 'trades' / 'trades.csv', index=False)
            
            # Save equity curve
            equity_df = pd.DataFrame({
                'equity': self.equity_curve,
                'drawdown': self.drawdown_curve
            })
            equity_df.to_csv(self.results_dir / 'equity' / 'equity.csv', index=False)
            
            # Save metrics
            with open(self.results_dir / 'metrics' / 'metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=4)
                
        except Exception as e:
            self.log.error(f"Error saving results: {str(e)}")
            
    def _generate_plots(self):
        """Generate performance plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve)
            plt.title('Equity Curve')
            plt.xlabel('Trade Number')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.savefig(self.results_dir / 'plots' / 'equity_curve.png')
            plt.close()
            
            # Drawdown
            plt.figure(figsize=(12, 6))
            plt.plot(self.drawdown_curve)
            plt.title('Drawdown')
            plt.xlabel('Trade Number')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.savefig(self.results_dir / 'plots' / 'drawdown.png')
            plt.close()
            
            # Trade distribution
            plt.figure(figsize=(12, 6))
            pnl_values = [trade.pnl for trade in self.trades]
            sns.histplot(pnl_values, bins=50)
            plt.title('Trade PnL Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(self.results_dir / 'plots' / 'pnl_distribution.png')
            plt.close()
            
        except Exception as e:
            self.log.error(f"Error generating plots: {str(e)}")
            
    def get_results(self) -> Dict:
        """Get backtest results."""
        return {
            'metrics': self.metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve
        } 