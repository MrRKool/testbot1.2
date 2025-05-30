import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
import optuna
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyParameters:
    entry_threshold: float
    exit_threshold: float
    stop_loss: float
    take_profit: float
    position_size: float
    max_positions: int

class StrategyTrainer:
    def __init__(self, 
                 historical_data: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0):
        """
        Initialize the strategy trainer.
        
        Args:
            historical_data: Dictionary of historical data by timeframe
            initial_balance: Initial balance for backtesting
        """
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.best_params = None
        self.best_performance = None
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for strategy optimization.
        """
        # Add your technical indicators here
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['atr'] = self._calculate_atr(df)
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def backtest_strategy(self, 
                         params: StrategyParameters,
                         data: pd.DataFrame) -> Dict:
        """
        Backtest a strategy with given parameters.
        
        Returns:
            Dictionary containing performance metrics
        """
        balance = self.initial_balance
        position = 0
        trades = []
        
        for i in range(len(data)):
            if i < 50:  # Skip warmup period
                continue
                
            current_price = data['close'].iloc[i]
            
            # Entry logic
            if position == 0 and self._check_entry_conditions(data.iloc[i], params):
                position = (balance * params.position_size) / current_price
                balance -= position * current_price
                trades.append({
                    'type': 'entry',
                    'price': current_price,
                    'size': position,
                    'timestamp': data.index[i]
                })
                
            # Exit logic
            elif position > 0 and self._check_exit_conditions(data.iloc[i], params):
                balance += position * current_price
                trades.append({
                    'type': 'exit',
                    'price': current_price,
                    'size': position,
                    'timestamp': data.index[i]
                })
                position = 0
                
        # Calculate performance metrics
        returns = pd.Series([t['price'] for t in trades]).pct_change()
        sharpe = self._calculate_sharpe_ratio(returns)
        max_dd = self._calculate_max_drawdown(returns)
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(trades),
            'final_balance': balance + (position * current_price if position > 0 else 0)
        }
        
    def _check_entry_conditions(self, 
                              data_point: pd.Series,
                              params: StrategyParameters) -> bool:
        """Check if entry conditions are met."""
        # Implement your entry logic here
        return (data_point['sma_20'] > data_point['sma_50'] and
                data_point['rsi'] < params.entry_threshold)
                
    def _check_exit_conditions(self,
                             data_point: pd.Series,
                             params: StrategyParameters) -> bool:
        """Check if exit conditions are met."""
        # Implement your exit logic here
        return (data_point['sma_20'] < data_point['sma_50'] or
                data_point['rsi'] > params.exit_threshold)
                
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        return drawdown.min()
        
    def optimize_strategy(self, 
                         n_trials: int = 100,
                         timeframe: str = '1h') -> Tuple[StrategyParameters, Dict]:
        """
        Optimize strategy parameters using Optuna.
        
        Args:
            n_trials: Number of optimization trials
            timeframe: Timeframe to use for optimization
            
        Returns:
            Tuple of (best parameters, best performance metrics)
        """
        data = self.historical_data[timeframe].copy()
        data = self.calculate_indicators(data)
        
        def objective(trial):
            params = StrategyParameters(
                entry_threshold=trial.suggest_float('entry_threshold', 20, 40),
                exit_threshold=trial.suggest_float('exit_threshold', 60, 80),
                stop_loss=trial.suggest_float('stop_loss', 0.01, 0.05),
                take_profit=trial.suggest_float('take_profit', 0.02, 0.1),
                position_size=trial.suggest_float('position_size', 0.1, 0.5),
                max_positions=trial.suggest_int('max_positions', 1, 5)
            )
            
            performance = self.backtest_strategy(params, data)
            return performance['sharpe_ratio']
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = StrategyParameters(
            entry_threshold=study.best_params['entry_threshold'],
            exit_threshold=study.best_params['exit_threshold'],
            stop_loss=study.best_params['stop_loss'],
            take_profit=study.best_params['take_profit'],
            position_size=study.best_params['position_size'],
            max_positions=study.best_params['max_positions']
        )
        
        self.best_performance = self.backtest_strategy(self.best_params, data)
        
        return self.best_params, self.best_performance 