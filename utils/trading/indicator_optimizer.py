import optuna
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from utils.trading.strategy import Strategy

# Stel logging in
logger = logging.getLogger("indicator_optimizer")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("indicator_optimizer.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Beschikbare indicatoren en hun parameter-ranges
def get_indicator_search_space():
    return {
        'RSI': {'period': (7, 30)},
        'MACD': {'fast_period': (5, 20), 'slow_period': (21, 40), 'signal_period': (5, 15)},
        'BB': {'period': (10, 30), 'std_dev': (1.5, 3.0)},
        'KVO': {'short_period': (20, 40), 'long_period': (40, 70)},
        'VI': {'period': (7, 30)},
        'CMF': {'period': (10, 30)},
        'DC': {'period': (10, 30)},
        'DPO': {'period': (10, 30)},
        'WILLR': {'period': (7, 30)},
        'STC': {'fast_period': (10, 30), 'slow_period': (30, 70)}
    }

# Evaluatiefunctie voor Optuna
def objective(trial, data: pd.DataFrame, base_config: Dict[str, Any], metric: str = 'sharpe'):
    indicator_space = get_indicator_search_space()
    indicators = {}
    # Indicatoren aan/uit en parameters kiezen
    for name, params in indicator_space.items():
        use_indicator = trial.suggest_categorical(f"use_{name}", [True, False])
        if use_indicator:
            indicators[name] = {}
            for param, rng in params.items():
                if isinstance(rng[0], int):
                    indicators[name][param] = trial.suggest_int(f"{name}_{param}", rng[0], rng[1])
                else:
                    indicators[name][param] = trial.suggest_float(f"{name}_{param}", rng[0], rng[1])
    # Config samenstellen
    config = base_config.copy()
    config['indicators'] = indicators
    config['use_rl'] = False  # Alleen klassieke strategie voor snelheid
    # Backtest uitvoeren
    try:
        strategy = Strategy(data.copy(), config)
        signals = []
        for i in range(1, len(data)):
            sig = strategy.generate_signal()
            signals.append(sig['action'])
        # Simpele PnL berekening
        returns = data['close'].pct_change().iloc[1:]
        positions = np.array([1 if s == 'BUY' else -1 if s == 'SELL' else 0 for s in signals])
        strat_returns = returns * positions
        pnl = np.nansum(strat_returns)
        sharpe = np.nan_to_num(np.mean(strat_returns) / (np.std(strat_returns) + 1e-8)) * np.sqrt(252)
        drawdown = np.nanmax(np.maximum.accumulate(np.cumsum(strat_returns)) - np.cumsum(strat_returns))
        # Score kiezen
        if metric == 'sharpe':
            score = sharpe
        elif metric == 'pnl':
            score = pnl
        elif metric == 'drawdown':
            score = -drawdown
        else:
            score = sharpe
        # Log trial
        logger.info(f"Trial: {indicators}, Sharpe: {sharpe:.3f}, PnL: {pnl:.3f}, Drawdown: {drawdown:.3f}")
        return score
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return -9999

def optimize_indicators(data: pd.DataFrame, base_config: Dict[str, Any], n_trials: int = 50, metric: str = 'sharpe') -> Dict[str, Any]:
    study = optuna.create_study(direction='maximize', study_name='indicator_optimization')
    study.optimize(lambda trial: objective(trial, data, base_config, metric), n_trials=n_trials)
    logger.info(f"Best trial: {study.best_trial.params}")
    # Bouw beste config
    best_config = base_config.copy()
    best_config['indicators'] = {}
    indicator_space = get_indicator_search_space()
    for name in indicator_space:
        if study.best_trial.params.get(f'use_{name}', False):
            best_config['indicators'][name] = {}
            for param in indicator_space[name]:
                best_config['indicators'][name][param] = study.best_trial.params[f'{name}_{param}']
    # Log en sla op
    with open('best_indicator_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Best config saved to best_indicator_config.json")
    return best_config

# Voorbeeld van gebruik:
if __name__ == "__main__":
    # Laad je data
    data = pd.read_csv('data/historical_prices.csv', index_col=0, parse_dates=True)
    # Basisconfig (vul aan met je eigen settings)
    base_config = {
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'indicators': {},
        'use_rl': False
    }
    # Optimaliseer
    best_config = optimize_indicators(data, base_config, n_trials=100, metric='sharpe')
    print("Beste indicatorconfiguratie:")
    print(json.dumps(best_config, indent=2)) 