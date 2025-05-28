import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from datetime import datetime
import yaml
import sys
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from functools import lru_cache
import numba
from numba import jit, prange
import mmap
import psutil
import json
from utils.risk_utils import get_risk_param

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.strategy.trading_strategy import TradingStrategy
from src.utils.data_loader import DataLoader
from src.risk.risk_manager import RiskManager

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    side: str  # 'long' or 'short'
    pnl: float
    fees: float
    slippage: float

class BacktestEngine:
    """Advanced backtesting engine with parallel processing and memory optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.backtest_config = config['backtest']
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.strategy = TradingStrategy(config)
        self.risk_manager = RiskManager(config)
        
        # Performance tracking
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest with parallel processing and memory optimization."""
        try:
            # Load data
            data = self.data_loader.load_historical_data(symbol, timeframe, start_date, end_date)
            if data.empty:
                self.logger.error(f"No data loaded for backtest: {symbol} {timeframe}")
                return {'trades': []}
                
            # Initialize components
            self.initial_capital = self.backtest_config['initial_capital']
            self.current_capital = self.initial_capital
            self.strategy.update_market_data(symbol, timeframe, data)
            self.risk_manager.initialize(self.initial_capital)
            
            # Make volume filter less strict for testing
            if 'symbols' in self.config and symbol in self.config['symbols']:
                self.config['symbols'][symbol]['min_volume'] = self.config['symbols'][symbol]['min_volume'] / 10
            
            # Process signals
            peak_equity = self.initial_capital
            for i in range(len(data)):
                if i < 50:  # Skip first 50 candles for indicator calculation
                    continue
                # Get current values
                current_price = data['close'].iloc[i]
                # Update the strategy's data for this bar
                self.strategy.update_market_data(symbol, timeframe, data.iloc[:i+1])
                # Get trading signal for this bar
                signal_info = self.strategy.get_trading_signal(symbol)
                current_signal = 0
                if signal_info['action'] == 'buy':
                    current_signal = 1
                elif signal_info['action'] == 'sell':
                    current_signal = -1
                # Process signal
                if current_signal != 0:
                    self._process_signal(current_signal, current_price, data.iloc[i])
                # Update positions
                self._update_positions(current_price, data.iloc[i])
                # Record equity
                self.equity_curve.append(self.current_capital)
                # --- DRAWNDOWN CHECK ---
                if self.current_capital > peak_equity:
                    peak_equity = self.current_capital
                current_drawdown = (peak_equity - self.current_capital) / peak_equity if peak_equity > 0 else 0
                max_drawdown = get_risk_param(self.config, symbol, 'max_drawdown')
                if max_drawdown is not None and current_drawdown >= max_drawdown:
                    self.logger.warning(f"Max drawdown bereikt: {current_drawdown*100:.2f}% >= {max_drawdown*100:.2f}%. Alle posities worden gesloten en de backtest wordt gestopt.")
                    self._close_all_positions(current_price, data.iloc[i])
                    break
            
            # Calculate performance metrics
            performance = self._calculate_performance(symbol)
            performance['trades'] = self.trades
            
            # Save results
            self._save_results(performance, symbol)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return {'trades': []}
            
    def _process_signal(self, signal: int, price: float, data: pd.Series):
        """Process trading signal with risk management."""
        try:
            # Check risk limits
            if not self.risk_manager._check_risk_limits(data):
                return
                
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=data['symbol'],
                price=price,
                volatility=data['atr'],
                account_balance=self.current_capital,
                signal_strength=abs(signal)
            )
            
            if position_size <= 0:
                return
                
            # Validate trade
            stop_loss = self._calculate_stop_loss(signal, price, data)
            take_profit = self._calculate_take_profit(signal, price, data)
            
            is_valid, reason = self.risk_manager.validate_trade(
                symbol=data['symbol'],
                side='long' if signal > 0 else 'short',
                size=position_size,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not is_valid:
                self.logger.info(f"Trade rejected: {reason}")
                return
                
            # Execute trade
            self._execute_trade(signal, position_size, price, stop_loss, take_profit, data)
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            
    def _calculate_stop_loss(self, signal: int, price: float, data: pd.Series) -> float:
        """Calculate stop loss level."""
        try:
            # Get stop loss percentage from config
            stop_loss_pct = get_risk_param(self.config, data['symbol'], 'stop_loss')
            if stop_loss_pct is None:
                stop_loss_pct = 0.015  # Default to 1.5%
            
            if signal > 0:  # Long
                return price * (1 - stop_loss_pct)
            else:  # Short
                return price * (1 + stop_loss_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return 0
            
    def _calculate_take_profit(self, signal: int, price: float, data: pd.Series) -> float:
        """Calculate take profit level."""
        try:
            # Get take profit percentage from config
            take_profit_pct = get_risk_param(self.config, data['symbol'], 'take_profit')
            if take_profit_pct is None:
                take_profit_pct = 0.03  # Default to 3%
            
            if signal > 0:  # Long
                return price * (1 + take_profit_pct)
            else:  # Short
                return price * (1 - take_profit_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return 0
            
    def _execute_trade(self, 
                      signal: int,
                      size: float,
                      price: float,
                      stop_loss: float,
                      take_profit: float,
                      data: pd.Series):
        """Execute trade with risk management."""
        try:
            # Calculate commission
            commission = price * size * get_risk_param(self.config, data['symbol'], 'commission')
            
            # Update capital
            self.current_capital -= commission
            
            # Record trade
            trade = {
                'timestamp': data.name,
                'symbol': data['symbol'],
                'side': 'long' if signal > 0 else 'short',
                'size': size,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'commission': commission
            }
            
            self.trades.append(trade)
            
            # Update positions
            if data['symbol'] not in self.positions:
                self.positions[data['symbol']] = []
                
            self.positions[data['symbol']].append(trade)
            
            self.logger.info(f"Trade executed: {trade}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            
    def _update_positions(self, current_price: float, data: pd.Series):
        """Update open positions with risk management."""
        try:
            for symbol, positions in self.positions.items():
                for position in positions:
                    # Calculate PnL
                    pnl = (current_price - position['price']) / position['price'] if position['side'] == 'long' else (position['price'] - current_price) / position['price']
                    
                    # Check if position should be closed
                    should_close, reason = self.risk_manager.update_position_risk(
                        symbol=symbol,
                        position=position,
                        current_price=current_price
                    )
                    
                    if should_close:
                        self._close_position(position, current_price, data)
                        self.logger.info(f"Position closed: {reason}")
                        
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            
    def _close_position(self, position: Dict[str, Any], current_price: float, data: pd.Series):
        """Close position with risk management."""
        try:
            # Calculate PnL
            pnl = (current_price - position['price']) / position['price'] if position['side'] == 'long' else (position['price'] - current_price) / position['price']
            
            # Calculate commission
            commission = current_price * position['size'] * get_risk_param(self.config, position['symbol'], 'commission')
            
            # Update capital
            self.current_capital += (position['size'] * current_price * (1 + pnl)) - commission
            
            # Remove position
            self.positions[data['symbol']].remove(position)
            
            self.logger.info(f"Position closed with PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            
    def _calculate_performance(self, symbol: str) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Calculate returns
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            # Calculate metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trades if t.get('pnl', 0) <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) <= 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            
            max_drawdown = self._calculate_max_drawdown()
            
            sharpe_ratio = self._calculate_sharpe_ratio(returns, symbol)
            sortino_ratio = self._calculate_sortino_ratio(returns, symbol)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance: {str(e)}")
            return {}
            
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            equity_curve = pd.Series(self.equity_curve)
            rolling_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - rolling_max) / rolling_max
            return abs(drawdowns.min())
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series, symbol: str) -> float:
        """Calculate Sharpe ratio."""
        try:
            risk_free_rate = get_risk_param(self.config, symbol, 'risk_free_rate') or 0
            excess_returns = returns - risk_free_rate
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
            
    def _calculate_sortino_ratio(self, returns: pd.Series, symbol: str) -> float:
        """Calculate Sortino ratio."""
        try:
            risk_free_rate = get_risk_param(self.config, symbol, 'risk_free_rate') or 0
            excess_returns = returns - risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            return np.sqrt(252) * excess_returns.mean() / downside_returns.std() if downside_returns.std() != 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0
            
    def _save_results(self, performance: Dict[str, Any], symbol: str):
        """Save backtest results."""
        try:
            # Create results directory
            results_dir = os.path.join('results', symbol)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save performance metrics
            performance_file = os.path.join(results_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(performance_file, 'w') as f:
                json.dump(performance, f, indent=4)
                
            # Save trades
            trades_file = os.path.join(results_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            
            # Save equity curve
            equity_file = os.path.join(results_dir, f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            pd.DataFrame({'equity': self.equity_curve}).to_csv(equity_file, index=False)
            
            self.logger.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def _close_all_positions(self, current_price: float, data: pd.Series):
        """Sluit alle open posities geforceerd af."""
        try:
            for symbol, positions in list(self.positions.items()):
                for position in positions[:]:
                    self._close_position(position, current_price, data)
            self.logger.info("Alle posities zijn gesloten vanwege max drawdown.")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}") 