import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from utils.trading.strategy import Strategy
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.logger import setup_logging, LogConfig
from tqdm import tqdm

class Backtest:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = YAMLLoader().load_config(config_path)
        self.setup_logging()
        self.strategy = Strategy(self.config)
        self.initial_capital = self.config['backtest']['initial_capital']
        self.commission_rate = self.config['backtest']['commission_rate']
        self.slippage_rate = self.config['backtest']['slippage_rate']
        self.results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }

    def setup_logging(self):
        log_config = LogConfig(
            log_dir='logs',
            log_level='INFO',
            log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            date_format='%Y-%m-%d %H:%M:%S',
            file_logging=True,
            console_logging=True
        )
        self.logger = setup_logging(log_config).get_logger('backtest')

    def load_data(self, symbol: str, tf: str, start_date: str, end_date: str, max_candles: int = 10000) -> pd.DataFrame:
        data_dir = Path(f'data/{symbol}')
        data_files = list(data_dir.glob(f'*{tf}*.csv'))
        if not data_files:
            raise FileNotFoundError(f"No data files found for {symbol} {tf}")
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("No timestamp column found in data")
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df[mask]
        if filtered_df.empty:
            raise ValueError(f"No data available for {symbol} {tf} between {start_date} and {end_date}")
        if len(filtered_df) > max_candles:
            filtered_df = filtered_df.iloc[-max_candles:]
        return filtered_df

    def calculate_metrics(self):
        """Calculate performance metrics."""
        trades = pd.DataFrame(self.results['trades'])
        if len(trades) == 0:
            return
        
        # Filter alleen trades met een 'pnl'
        trades = trades[trades['pnl'].notna()]
        if len(trades) == 0:
            return
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        self.results['metrics'] = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        equity_curve = pd.Series(self.results['equity_curve'])
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min())

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity curve."""
        equity_curve = pd.Series(self.results['equity_curve'])
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()

    def run_multi_tf(self, symbol: str, start_date: str, end_date: str):
        self.logger.info(f"Starting multi-timeframe backtest for {symbol} from {start_date} to {end_date}")
        # Laad data voor alle timeframes
        data_1m = self.load_data(symbol, '1m', start_date, end_date)
        data_5m = self.load_data(symbol, '5m', start_date, end_date)
        data_1h = self.load_data(symbol, '1h', start_date, end_date)
        # Bereken indicatoren op hogere timeframes
        strat_5m = Strategy(self.config)
        strat_1h = Strategy(self.config)
        strat_5m.update_market_data(data_5m)
        strat_1h.update_market_data(data_1h)
        # Sla trends op per candle
        data_5m['trend'] = (data_5m['EMA_short'] > data_5m['EMA_medium']) & (data_5m['EMA_medium'] > data_5m['EMA_long'])
        data_1h['trend'] = (data_1h['EMA_short'] > data_1h['EMA_medium']) & (data_1h['EMA_medium'] > data_1h['EMA_long'])
        data_5m['trend_down'] = (data_5m['EMA_short'] < data_5m['EMA_medium']) & (data_5m['EMA_medium'] < data_5m['EMA_long'])
        data_1h['trend_down'] = (data_1h['EMA_short'] < data_1h['EMA_medium']) & (data_1h['EMA_medium'] < data_1h['EMA_long'])
        # Backtest loop op 1m
        position = None
        entry_price = 0
        current_capital = self.initial_capital
        window = 50
        for i in tqdm(range(len(data_1m)), desc=f"Backtesting {symbol} multi-tf 1m"):
            current_data = data_1m.iloc[max(0, i-window+1):i+1].copy()
            ts = current_data.index[-1]
            # Zoek de laatste 5m en 1h candle die <= ts is
            tf5_idx = data_5m.index.get_indexer([ts], method='pad')[0] if ts >= data_5m.index[0] else 0
            tf1h_idx = data_1h.index.get_indexer([ts], method='pad')[0] if ts >= data_1h.index[0] else 0
            tf5_trend = data_5m.iloc[tf5_idx]['trend']
            tf1h_trend = data_1h.iloc[tf1h_idx]['trend']
            tf5_trend_down = data_5m.iloc[tf5_idx]['trend_down']
            tf1h_trend_down = data_1h.iloc[tf1h_idx]['trend_down']
            # Update 1m strategie
            self.strategy.update_market_data(current_data)
            signal = self.strategy.generate_signal(current_data)
            # Multi-tf filter: alleen long als alle trends up, alleen short als alle trends down
            if signal:
                if signal['type'] == 'long' and (tf5_trend and tf1h_trend):
                    current_price = current_data.iloc[-1]['close']
                    if position is None:
                        position = 'long'
                        entry_price = current_price
                        commission = current_price * self.commission_rate
                        slippage = current_price * self.slippage_rate
                        current_capital -= (commission + slippage)
                        self.results['trades'].append({
                            'entry_time': ts,
                            'entry_price': entry_price,
                            'position': position,
                            'commission': commission,
                            'slippage': slippage,
                            'pnl': None
                        })
                    elif position == 'short':
                        exit_price = current_price
                        commission = exit_price * self.commission_rate
                        slippage = exit_price * self.slippage_rate
                        pnl = (entry_price - exit_price) / entry_price
                        pnl -= (commission + slippage) / current_capital
                        current_capital *= (1 + pnl)
                        self.results['trades'][-1].update({
                            'exit_time': ts,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'final_capital': current_capital
                        })
                        position = 'long'
                        entry_price = current_price
                elif signal['type'] == 'short' and (tf5_trend_down and tf1h_trend_down):
                    current_price = current_data.iloc[-1]['close']
                    if position is None:
                        position = 'short'
                        entry_price = current_price
                        commission = current_price * self.commission_rate
                        slippage = current_price * self.slippage_rate
                        current_capital -= (commission + slippage)
                        self.results['trades'].append({
                            'entry_time': ts,
                            'entry_price': entry_price,
                            'position': position,
                            'commission': commission,
                            'slippage': slippage,
                            'pnl': None
                        })
                    elif position == 'long':
                        exit_price = current_price
                        commission = exit_price * self.commission_rate
                        slippage = exit_price * self.slippage_rate
                        pnl = (exit_price - entry_price) / entry_price
                        pnl -= (commission + slippage) / current_capital
                        current_capital *= (1 + pnl)
                        self.results['trades'][-1].update({
                            'exit_time': ts,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'final_capital': current_capital
                        })
                        position = 'short'
                        entry_price = current_price
            self.results['equity_curve'].append(current_capital)
        self.calculate_metrics()
        self.log_results()
        return self.results

    def log_results(self):
        """Log backtest results."""
        metrics = self.results['metrics']
        if not metrics or 'total_trades' not in metrics:
            self.logger.info("No trades executed during backtest")
            return
            
        self.logger.info("Backtest Results:")
        self.logger.info(f"Total Trades: {metrics['total_trades']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"Average Win: {metrics['avg_win']:.2%}")
        self.logger.info(f"Average Loss: {metrics['avg_loss']:.2%}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol = 'BTCUSDT'
    start_date = '2024-01-01'
    end_date = '2024-02-01'
    print(f"\n=== Multi-timeframe Backtest voor {symbol} (1m entry, 5m+1h trend) ===")
    backtest = Backtest()
    results = backtest.run_multi_tf(symbol, start_date, end_date)
    results_dir = Path('results') / symbol
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results['trades'])
    results_df.to_csv(results_dir / f'{symbol}_trades_multiTF_{timestamp}.csv', index=False)
    equity_df = pd.DataFrame({'equity': results['equity_curve']})
    equity_df.to_csv(results_dir / f'{symbol}_equity_multiTF_{timestamp}.csv', index=False)
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_json(results_dir / f'{symbol}_metrics_multiTF_{timestamp}.json', orient='records')
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title(f'{symbol} Equity Curve Multi-TF')
    plt.xlabel('Trade Number')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.savefig(results_dir / f'{symbol}_equity_curve_multiTF_{timestamp}.png')
    plt.close()
    metrics = results['metrics']
    if not metrics or 'total_trades' not in metrics:
        print(f"Geen trades gevonden voor {symbol} (multi-tf).")
    else:
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Win: {metrics['avg_win']:.2%}")
        print(f"Average Loss: {metrics['avg_loss']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

if __name__ == '__main__':
    main() 