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
from src.ai.ai_learning import MultiTimeframeAnalyzer

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
        self.strategy = TradingStrategy(config)  # Initialize strategy first
        self.risk_manager = RiskManager(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(config)
        
        # Performance tracking
        self.initial_capital = config['backtest']['initial_capital']
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Additional attributes
        self.commission_rate = config['backtest']['commission_rate']
        self.slippage_rate = config['backtest']['slippage_rate']
        self.start_date = pd.to_datetime(config['backtest']['start_date'])
        self.end_date = pd.to_datetime(config['backtest']['end_date'])
        self.timeframes = config['backtest']['timeframes']
        self.data_dir = config['backtest']['data_dir']
        
        # Initialize data containers for each timeframe
        self.data = {}
        self.indicators = {}
        self.signals = {}
        
        # Load historical data for each timeframe
        self._load_historical_data()
        
        # Assign data to strategy
        self.strategy.data['BTCUSDT'] = {}
        for timeframe in self.timeframes:
            self.strategy.data['BTCUSDT'][timeframe] = self.data[timeframe].copy()
            # Calculate indicators immediately after assigning data
            self.indicators[timeframe] = self.strategy.calculate_indicators('BTCUSDT', timeframe)
        
    def _load_historical_data(self):
        """Load historical data for all timeframes"""
        for timeframe in self.timeframes:
            file_path = os.path.join(self.data_dir, f"BTCUSDT_{timeframe}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                self.data[timeframe] = df
            else:
                raise FileNotFoundError(f"Historical data not found for timeframe {timeframe}")
    
    def _calculate_indicators(self):
        """Calculate indicators for all timeframes"""
        for timeframe in self.timeframes:
            df = self.data[timeframe]
            self.indicators[timeframe] = self.strategy.calculate_indicators('BTCUSDT', timeframe)
            # Ensure ATR is available in the indicators DataFrame
            if 'atr' not in self.indicators[timeframe]:
                self.indicators[timeframe]['atr'] = df['atr']
    
    def _generate_signals(self):
        """Generate trading signals using multi-timeframe analysis"""
        # Initialize signal DataFrame for each timeframe
        for timeframe in self.timeframes:
            self.signals[timeframe] = pd.DataFrame(index=self.data[timeframe].index)
            self.signals[timeframe]['signal'] = 0
        
        # Generate signals for each timeframe
        for timeframe in self.timeframes:
            df = self.data[timeframe]
            indicators = self.indicators[timeframe]
            # Uitgebreide debug info
            print(f"[DEBUG] data shape voor {timeframe}: {df.shape}")
            print(f"[DEBUG] data columns voor {timeframe}: {list(df.columns)}")
            print(f"[DEBUG] data head voor {timeframe}:\n{df.head()}")
            print(f"[DEBUG] indicators shape voor {timeframe}: {indicators.shape}")
            print(f"[DEBUG] indicators columns voor {timeframe}: {list(indicators.columns)}")
            print(f"[DEBUG] indicators head voor {timeframe}:\n{indicators.head()}")
            # Calculate signals using strategy
            signals = self.strategy.generate_signals(data=df, indicators=indicators)
            self.signals[timeframe]['signal'] = signals
        
        # Combine signals from different timeframes
        self._combine_timeframe_signals()
    
    def _combine_timeframe_signals(self):
        """Combine signals from different timeframes with weights"""
        # Define weights for each timeframe
        weights = {
            '1m': 0.1,
            '5m': 0.2,
            '1h': 0.4,
            '4h': 0.3
        }
        
        # Initialize combined signals DataFrame
        combined_signals = pd.DataFrame(index=self.data['1h'].index)
        combined_signals['signal'] = 0
        
        # Combine signals with weights
        for timeframe, weight in weights.items():
            # Resample signals to 1h timeframe using 'h' instead of 'H'
            resampled_signals = self.signals[timeframe]['signal'].resample('1h').last()
            combined_signals['signal'] += resampled_signals * weight
        
        # Normalize combined signals
        combined_signals['signal'] = combined_signals['signal'] / sum(weights.values())
        
        # Apply threshold for final signal
        combined_signals['final_signal'] = 0
        combined_signals.loc[combined_signals['signal'] > 0.6, 'final_signal'] = 1
        combined_signals.loc[combined_signals['signal'] < -0.6, 'final_signal'] = -1
        
        self.signals['combined'] = combined_signals
    
    def run(self):
        """Run the backtest with multi-timeframe analysis"""
        print("Starting backtest...")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Timeframes: {self.timeframes}")
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Generate signals
        self._generate_signals()
        
        # Run backtest using combined signals
        self._run_backtest()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        return self.metrics
    
    def _run_backtest(self):
        """Run the backtest using combined signals"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        # Use 1h data for backtest execution
        data = self.data['1h']
        signals = self.signals['combined']
        
        for timestamp, row in data.iterrows():
            if timestamp < self.start_date or timestamp > self.end_date:
                continue
                
            signal = signals.loc[timestamp, 'final_signal']
            price = row['close']
            
            # Handle position entry
            if signal != 0 and position == 0:
                position = signal
                entry_price = price
                capital *= (1 - self.commission_rate)
                
            # Handle position exit
            elif (signal == -position or 
                  (position > 0 and price <= entry_price * (1 - self.config['trading']['risk']['stop_loss'])) or
                  (position < 0 and price >= entry_price * (1 + self.config['trading']['risk']['stop_loss'])) or
                  (position > 0 and price >= entry_price * (1 + self.config['trading']['risk']['take_profit'])) or
                  (position < 0 and price <= entry_price * (1 - self.config['trading']['risk']['take_profit']))):
                
                # Calculate profit/loss with check for zero entry_price
                if entry_price != 0:
                    pnl = position * (price - entry_price) / entry_price
                else:
                    pnl = 0
                    
                capital *= (1 + pnl) * (1 - self.commission_rate)
                
                # Record trade
                self.trades.append({
                    'entry_time': entry_price,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position': position,
                    'pnl': pnl,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': capital
            })
            
            # Update positions
            self._update_positions(price, row)
            
            # Check drawdown
            if capital > self.initial_capital:
                peak_equity = capital
                current_drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0
                max_drawdown = get_risk_param(self.config, 'BTCUSDT', 'max_drawdown')
                if max_drawdown is not None and current_drawdown >= max_drawdown:
                    self.logger.warning(f"Max drawdown bereikt: {current_drawdown*100:.2f}% >= {max_drawdown*100:.2f}%. Alle posities worden gesloten en de backtest wordt gestopt.")
                    self._close_all_positions(price, row)
                    break
            
        # Calculate performance metrics
        self.metrics = self._calculate_performance()
        
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
            
    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Calculate returns
            returns = pd.Series([t['pnl'] for t in self.trades]).pct_change().dropna()
            
            # Calculate metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] <= 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            
            max_drawdown = self._calculate_max_drawdown()
            
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
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
            equity_curve = pd.Series([t['equity'] for t in self.equity_curve])
            rolling_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - rolling_max) / rolling_max
            return abs(drawdowns.min())
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Use default risk_free_rate of 0.02 (2%) if not specified
            risk_free_rate = get_risk_param(self.config, 'BTCUSDT', 'risk_free_rate') or 0.02
            excess_returns = returns - risk_free_rate
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
            
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            # Use default risk_free_rate of 0.02 (2%) if not specified
            risk_free_rate = get_risk_param(self.config, 'BTCUSDT', 'risk_free_rate') or 0.02
            excess_returns = returns - risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            return np.sqrt(252) * excess_returns.mean() / downside_returns.std() if downside_returns.std() != 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0
            
    def _save_results(self):
        """Save backtest results."""
        try:
            # Create results directory
            results_dir = os.path.join('results', 'BTCUSDT')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save performance metrics
            performance_file = os.path.join(results_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(performance_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
                
            # Save trades
            trades_file = os.path.join(results_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            
            # Save equity curve
            equity_file = os.path.join(results_dir, f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            pd.DataFrame({'equity': [t['equity'] for t in self.equity_curve]}).to_csv(equity_file, index=False)
            
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

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Calculate returns with explicit fill_method=None
            returns = pd.Series([t['pnl'] for t in self.trades]).pct_change(fill_method=None).dropna()
            
            # Calculate metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] <= 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            
            max_drawdown = self._calculate_max_drawdown()
            
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
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